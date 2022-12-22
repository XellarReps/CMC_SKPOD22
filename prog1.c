#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

int RecvNumberFrom(int coordinates[2], MPI_Comm cart) {
    MPI_Status status;
    int rank, number;
    MPI_Cart_rank(cart, coordinates, &rank);
    MPI_Recv(&number, 1, MPI_INT, rank, 0, cart, &status);
    return number;
}

void ISendNumberTo(int coordinates[2], int number, MPI_Comm cart) {
    MPI_Request request;
    int rank;
    MPI_Cart_rank(cart, coordinates, &rank);
    MPI_Isend(&number, 1, MPI_INT, rank, 0, cart, &request);
}

int Master(int coordinates[2], int number, MPI_Comm cart) {
    int up_coordinates[2] = {coordinates[0], coordinates[1] + 1};
    int right_coordinates[2] = {coordinates[0] + 1, coordinates[1]};

    int up_number = RecvNumberFrom(up_coordinates, cart);
    int right_number = RecvNumberFrom(right_coordinates, cart);

    int max_number = max(up_number, right_number);
    max_number = max(max_number, number);
    printf("Max number %d\n", max_number); fflush(stdout);

    ISendNumberTo(up_coordinates, max_number, cart);
    ISendNumberTo(right_coordinates, max_number, cart);
    return max_number;
}

int DownRightProcess(int coordinates[2], int number, MPI_Comm cart) {
    int up_coordinates[2] = {coordinates[0], coordinates[1] + 1};
    int left_coordinates[2] = {coordinates[0] - 1, coordinates[1]};

    int up_number = RecvNumberFrom(up_coordinates, cart);

    int max_number = max(number, up_number);

    ISendNumberTo(left_coordinates, max_number, cart);

    max_number = RecvNumberFrom(left_coordinates, cart);
    ISendNumberTo(up_coordinates, max_number, cart);
    return max_number;
}

int DownProcesses(int coordinates[2], int number, MPI_Comm cart) {
    int up_coordinates[2] = {coordinates[0], coordinates[1] + 1};
    int right_coordinates[2] = {coordinates[0] + 1, coordinates[1]};
    int left_coordinates[2] = {coordinates[0] - 1, coordinates[1]};

    int up_number = RecvNumberFrom(up_coordinates, cart);
    int right_number = RecvNumberFrom(right_coordinates, cart);

    int max_number = max(up_number, right_number);
    max_number = max(max_number, number);

    ISendNumberTo(left_coordinates, max_number, cart);


    max_number = RecvNumberFrom(left_coordinates, cart);
    ISendNumberTo(up_coordinates, max_number, cart);
    ISendNumberTo(right_coordinates, max_number, cart);
    return max_number;
}

int UpProcesses(int coordinates[2], int number, MPI_Comm cart) {
    int down_coordinates[2] = {coordinates[0], coordinates[1] - 1};
    ISendNumberTo(down_coordinates, number, cart);

    int max_number = RecvNumberFrom(down_coordinates, cart);
    return max_number;
}

int InternalProcesses(int coordinates[2], int number, MPI_Comm cart) {
    int up_coordinates[2] = {coordinates[0], coordinates[1] + 1};
    int down_coordinates[2] = {coordinates[0], coordinates[1] - 1};

    int up_number = RecvNumberFrom(up_coordinates, cart);

    int max_number = max(number, up_number);

    ISendNumberTo(down_coordinates, max_number, cart);

    max_number = RecvNumberFrom(down_coordinates, cart);
    ISendNumberTo(up_coordinates, max_number, cart);
    return max_number;
}

int main(int argc, char *argv[]) {
	int size;
    const int dims[2] = {4, 4};
	const int periods[2] = {0, 0};
    MPI_Comm cart;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Cart_create(MPI_COMM_WORLD, 2 /*ndims*/, dims, periods, 0 /*reorder*/, &cart);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int coordinates[2];
	MPI_Cart_coords(cart, rank, 2, coordinates);

    int number = rank;
    printf("(%d, %d): %d\n", coordinates[0], coordinates[1], number); fflush(stdout);

    int max_number;
	if (coordinates[0] == 0 && coordinates[1] == 0) {
        max_number =Master(coordinates, number, cart);
    } else if (coordinates[0] == 3 && coordinates[1] == 0) {
        max_number = DownRightProcess(coordinates, number, cart);
    } else if (coordinates[1] == 0) {
        max_number = DownProcesses(coordinates, number, cart);
    } else if (coordinates[1] == 3) {
        max_number = UpProcesses(coordinates, number, cart);
    } else {
        max_number = InternalProcesses(coordinates, number, cart);
    }

    printf("Max number (%d, %d): %d\n", coordinates[0], coordinates[1], max_number); fflush(stdout);

    MPI_Finalize();
    return 0;
}