#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) { 
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	double m_size_double = pow(10.0, (double) atoi(argv[1])/10);
	//double m_size_double = atoi(argv[1]);
	int message_size = (int)m_size_double;
	char *data = malloc(message_size);
	int count = 1e7/message_size;
	if (count  < 100) count = 100;
	double t = MPI_Wtime();

	for (int i = 0; i < count; i++) {
	if (rank == 0) { 
		MPI_Send(data, message_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
		MPI_Recv(data, message_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	} else if (rank == 1) {
		MPI_Recv(data, message_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(data, message_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
	} 
		
	}
	t = MPI_Wtime() - t;	
	double ts = t * 0.5/count; //s
	t = t * 1e6 * 0.5/count; //mcs
	double band = message_size/ts/1024/1024; //Mbyte/s 
	
	FILE *fp;
	char name[] = "latency.txt";
	if ((fp = fopen(name, "a")) == NULL) { 
		printf("Can't find file");
		return 1;
	}

	if (rank == 0) fprintf(fp, "%d %f %f\n", message_size, t, band);
	fclose(fp);
	
	free(data);
	MPI_Finalize();
	return 0;

}
