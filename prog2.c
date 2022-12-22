#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <mpi.h>

#define  READ_MATRIX_FROM_FILE (0)

#define  Max(a,b) ((a)>(b)?(a):(b))

void
init(int n, float *A)
{
	for (int i = 0; i < n; i++) {
	    for (int j = 0; j < n; j++) {
	        for (int k = 0; k < n; k++) {
	            if (i == 0 || i == n - 1 || j == 0 || j == n - 1 || k == 0 || k == n - 1) {
	                A[i * n * n + n * j + k] = 0.;
                } else {
                    A[i * n * n + n * j + k] = (4. + i + j + k);
                }
            }
        }
    }
}

float
relax_mpi(int n, float *A, int myrank, float *myA1, float *myA2, int *cnt1, int *disp1, int *cnt2, int *disp2, MPI_Datatype COLRES, MPI_Comm comm)
{
    static int count = 0;
    ++count;

    int myn = cnt1[myrank];
    int arr_size = myn * n;
    float eps = 0.;
    float eps2 = 0.;

    if (myrank == 0) {
        FILE *input = fopen("A.txt", "r");
        for (int i = 0; i < n * n * n; ++i) {
            fscanf(input, "%f", &A[i]);
        }
        fclose(input);
    }

    if (count == 1 && myrank == 1) {
        printf("%d: failed on %d launch\n", myrank, count); fflush(stdout);
        raise(SIGKILL);
    }

    int err = MPI_Scatterv(A, cnt1, disp1, COLRES, myA1, arr_size, MPI_FLOAT, 0, comm);
    if (err != MPI_SUCCESS) {
        printf("%d: get fail on %d launch\n", myrank, count); fflush(stdout);
        return -1;
    }

    int cnt = 0;
    for (int j = 0; j < myn; j++) {
        for (int i = 0; i < n; i++) {
            if (i != 0 && i != n - 1) {
                myA1[cnt] = (myA1[cnt - 1] + myA1[cnt + 1]) / 2.;
            }
            cnt++;
        }
    }


    if (count == 2 && (myrank == 1 || myrank == 2)) {
        printf("%d: failed on %d launch\n", myrank, count); fflush(stdout);
        raise(SIGKILL);
    }

    err = MPI_Gatherv(myA1, arr_size, MPI_FLOAT, A, cnt1, disp1, COLRES, 0, comm);
    if (err != MPI_SUCCESS) {
        printf("%d: get fail on %d launch\n", myrank, count); fflush(stdout);
        return -1;
    }

    arr_size = cnt2[myrank];
    myn = arr_size / (n * n);


    if (count == 3 && myrank == 3) {
        printf("%d: failed on %d launch\n", myrank, count); fflush(stdout);
        raise(SIGKILL);
    }

    err = MPI_Scatterv(A, cnt2, disp2, MPI_FLOAT, myA2, arr_size, MPI_FLOAT, 0, comm);
    if (err != MPI_SUCCESS) {
        printf("%d: get fail on %d launch\n", myrank, count); fflush(stdout);
        return -1;
    }

    cnt = 0;
    for (int i = 0; i < myn; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                if (j != 0 && j != n - 1) {
                    myA2[cnt] = (myA2[cnt - n] + myA2[cnt + n]) / 2.;
                }
                cnt++;
            }
        }
    }
    cnt = 0;
    for (int i = 0; i < myn; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                if (k != 0 && k != n - 1) {
                    float e = myA2[cnt];
                    myA2[cnt] = (myA2[cnt - 1] + myA2[cnt + 1]) / 2.;
                    eps = Max(eps, fabs(e - myA2[cnt]));
                }
                cnt++;
            }
        }
    }

    if (count == 4 && myrank == 2) {
        printf("%d: failed on %d launch\n", myrank, count); fflush(stdout);
        raise(SIGKILL);
    }

    err = MPI_Gatherv(myA2, arr_size, MPI_FLOAT, A, cnt2, disp2, MPI_FLOAT, 0, comm);
    if (err != MPI_SUCCESS) {
        printf("%d: get fail on %d launch\n", myrank, count); fflush(stdout);
        return -1;
    }

    err = MPI_Reduce(&eps, &eps2, 1, MPI_FLOAT, MPI_MAX, 0, comm);
    if (err != MPI_SUCCESS) {
        printf("%d: get fail on %d launch\n", myrank, count); fflush(stdout);
        return -1;
    }

    err = MPI_Bcast(&eps2, 1, MPI_FLOAT, 0, comm);
    if (err != MPI_SUCCESS) {
        printf("%d: get fail on %d launch\n", myrank, count); fflush(stdout);
        return -1;
    }


    if (myrank == 0) {
        FILE *output = fopen("A.txt", "w");
        for (int i = 0; i < arr_size; ++i) {
            fprintf(output, "%f", A[i]);
        }
        fclose(output);
    }

    return eps2;
}

void
wrap(
    int n, float * A, int itmax, float mineps, MPI_Comm comm)
{
    int size;
    MPI_Comm_size(comm, &size);

    int myrank;
    MPI_Comm_rank(comm, &myrank);


    int *cnt1, * disp1;
    cnt1 = calloc(size, sizeof(*cnt1));
    disp1 = calloc(size, sizeof(*disp1));
    int * cnt2, * disp2;
    cnt2 = calloc(size, sizeof(*cnt2));
    disp2 = calloc(size, sizeof(*disp2));
    MPI_Datatype COL, COLRES;
    MPI_Type_vector(n, 1, n * n, MPI_FLOAT, &COL);
    MPI_Type_commit(&COL);
    MPI_Type_create_resized(COL, 0, sizeof(float), &COLRES);
    MPI_Type_commit(&COLRES);

    cnt1[0] = (n * n / size + (0 < (n * n) % size));
    disp1[0] = 0;
    for (int i = 1; i < size; i++) {
        cnt1[i] = cnt1[i - 1];
        if (i == (n * n) % size) {
            cnt1[i]--;
        }
        disp1[i] = disp1[i - 1] + cnt1[i - 1];
    }
    cnt2[0] = (n / size + (0 < n % size)) * n * n;
    disp2[0] = 0;
    for (int i = 1; i < size; i++) {
        cnt2[i] = cnt2[i - 1];
        if (i == n % size) {
            cnt2[i] -= n * n;
        }
        disp2[i] = disp2[i - 1] + cnt2[i - 1];
    }
    int arr_size = cnt1[myrank] * n;
    float *myA1 = calloc(arr_size, sizeof(*myA1));
    arr_size = cnt2[myrank];
    float *myA2 = calloc(arr_size, sizeof(*myA2));
    for(int it = 0; it < itmax; it++)
    {
	    float eps = relax_mpi(n, A, myrank, myA1, myA2, cnt1, disp1, cnt2, disp2, COLRES, comm);
        if (eps < 0) {
            // repeat this iteration with new comm
            free(myA1);
            free(myA2);
            MPI_Type_free(&COL);
            MPI_Type_free(&COLRES);
            free(cnt1);
            free(disp1);
            free(cnt2);
            free(disp2);

            MPIX_Comm_shrink(comm, &comm);

            wrap(n, A, itmax, mineps, comm);
            return;
        }
	    if (eps < mineps) {
            break;
        }
    }
    free(myA1);
    free(myA2);
    MPI_Type_free(&COL);
    MPI_Type_free(&COLRES);
    free(cnt1);
    free(disp1);
    free(cnt2);
    free(disp2);
}

void
verify(int n, float *A)
{
	float s = 0.;
	for (int i = 1; i < n - 1; i++) {
	    for (int j = 1; j < n - 1; j++) {
	        for (int k = 1; k < n - 1; k++) {
		        s += A[i * n * n + n * j + k] * (i + 1) * (j + 1) * (k + 1) / (n * n * n);
	        }
        }
    }
	printf("S = %f\n", s);
}

int main (int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    const float mineps = 0.1e-7;
    const int itmax = 100;
    int numproc = 64;
    int myrank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);
    MPI_Comm_rank(comm, &myrank);
    FILE * f = NULL;
    if (myrank == 0) {
        f = fopen("output_proc64", "w");
    }
    int n = 16;
    double start, end;
    float *A;
    if (myrank == 0) {
        A = calloc(n * n * n, sizeof(*A));
        if (!READ_MATRIX_FROM_FILE) {
            init(n, A);
            FILE *output = fopen("A.txt", "w");
            for (int i = 0; i < n * n * n; ++i) {
                fprintf(output, "%f", A[i]);
            }
            fclose(output);
        }
    }
    MPI_Barrier(comm);
    if (myrank == 0) {
        start = MPI_Wtime();
    }

    wrap(n, A, itmax, mineps, comm);

    MPI_Barrier(comm);
    if (myrank == 0) {
        end = MPI_Wtime();
        fprintf(f, "%d\t%d\t%lf\n", n, numproc, end - start);
        // verify(n, A); // we comment so as not to waste time on checking as it has already been checked
        free(A);
    }
    if (myrank == 0) {
        fclose(f);
    }
    MPI_Finalize();
	return 0;
}