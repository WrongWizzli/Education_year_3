#define MINI_DATASET
#include <mpi.h>
#include "gemm.h"

double bench_t_start, bench_t_end;

static double rtclock() {
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
        printf ("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start() {
    bench_t_start = rtclock ();
}

void bench_timer_stop() {
    bench_t_end = rtclock ();
}

double bench_timer_print() {
    printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
    return bench_t_end - bench_t_start;
}

static void init_array(int ni, int nj, int nk, double *alpha, double *beta, double C[ni][nj], double A[ni][nk], double B[nk][nj]) {
    int i, j;

    *alpha = 1.5;
    *beta = 1.2;
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            C[i][j] = (double) ((i * j + 1) % ni) / ni;
        }
    }
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nk; j++) {
            A[i][j] = (double) (i * (j + 1) % nk) / nk;
        }
    }
    for (i = 0; i < nk; i++) {
        for (j = 0; j < nj; j++) {
            B[i][j] = (double) (i * (j + 2) % nj) / nj;
        }
    }
}

static void print_array(int ni, int nj, double C[ni][nj]) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "C\n");
  for (i = 0; i < ni; i++)  {
      for (j = 0; j < nj; j++) {
          printf("%0.2lf\t", C[i][j]);
      }
      printf("\n");
  }
  fprintf(stderr, "\nend   dump: %s\n", "C");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_gemm(int ni, int nj, int nk, double alpha, double beta, double C[ni][nj], double A[ni][nk], double B[nk][nj]) {
    int i, j, k;
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++)
            C[i][j] *= beta;
        for (k = 0; k < nk; k++) {
            for (j = 0; j < nj; j++)
                C[i][j] += alpha * A[i][k] * B[k][j];
        }
    }
}

int main(int argc, char** argv) {

    int ni = NI;
    int nj = NJ;
    int nk = NK;

    double alpha;
    double beta;
    double C[ni][nj];
    double A[ni][nk];
    double B[nk][nj];

    int task_id, N;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
    MPI_Comm_size(MPI_COMM_WORLD, &N);

    if (!taskid) {

        int offset = 0, rows = ni / (N - 1);;

        for (dst=1; dst < N; dst++) {
            MPI_Send(&offset, 1, MPI_INT, dst, 1, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dst, 1, MPI_COMM_WORLD);
            MPI_Send(&a[offset][0], rows * ni, MPI_DOUBLE, dst, 1, MPI_COMM_WORLD);
            MPI_Send(&b, nj * nk, MPI_DOUBLE, dst, 1, MPI_COMM_WORLD);
            offset += rows;
        }

        int i, j, k;
        for (i = 1; i < N; i++) {
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset][0], rows*N, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
        }

        gettimeofday(&stop, 0);

        printf("Here is the result matrix:\n");
        for (i=0; i<N; i++) {
        for (j=0; j<N; j++)
            printf("%6.2f   ", c[i][j]);
        printf ("\n");
        }

        fprintf(stdout,"Time = %.6f\n\n",
            (stop.tv_sec+stop.tv_usec*1e-6)-(start.tv_sec+start.tv_usec*1e-6));

    }

    /*---------------------------- worker----------------------------*/
    if (taskid > 0) {
        source = 0;
        MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&a, rows*N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&b, N*N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);

        /* Matrix multiplication */
        for (k=0; k<N; k++)
        for (i=0; i<rows; i++) {
            c[i][k] = 0.0;
            for (j=0; j<N; j++)
            c[i][k] = c[i][k] + a[i][j] * b[j][k];
        }


        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&c, rows*N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    }