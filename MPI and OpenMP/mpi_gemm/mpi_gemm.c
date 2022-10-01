#define MINI_DATASET
#include "mpigemm.h"
MPI_Status status;

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



int main(int argc, char** argv) {

    int rows, loop_row, extra_rows, task_id, N, offset;
    int dst, i, j, k;

    int ni = NI;
    int nj = NJ;
    int nk = NK;

    double alpha;
    double beta;
    double C[ni][nj];
    double A[ni][nk];
    double B[nk][nj];

    init_array(ni, nj, nk, &alpha, &beta, C, A, B);

    double avg_exec_time = 0.0;

    bench_timer_start();

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
    MPI_Comm_size(MPI_COMM_WORLD, &N);

    if (!task_id) {


        printf("Enter number of iterations:\n");
        printf("%d %d\n", N, task_id);
        printf("%d %d\n", ni, nj);


        rows = ni / (N - 1);
        extra_rows = ni % (N - 1);

        offset = 0;

        for (dst = 1; dst < N; ++dst) {
            if (dst <= extra_rows) {
                loop_row = rows + 1;
            } else {
                loop_row = rows;
            }
            loop_row = (dst <= extra_rows) ? rows + 1 : rows;
            MPI_Send(&offset, 1, MPI_INT, dst, 1, MPI_COMM_WORLD);
            MPI_Send(&loop_row, 1, MPI_INT, dst, 1, MPI_COMM_WORLD);
            MPI_Send(&A[offset][0], loop_row * nk, MPI_DOUBLE, dst, 1, MPI_COMM_WORLD);
            MPI_Send(&B, nk * nj, MPI_DOUBLE, dst, 1, MPI_COMM_WORLD);

            offset += rows;
        }

        for (j = 1; j < N; ++j) {

            MPI_Recv(&offset, 1, MPI_INT, j, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&loop_row, 1, MPI_INT, j, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&C[offset][0], loop_row * nj, MPI_DOUBLE, j, 2, MPI_COMM_WORLD, &status);

        }
    } else if (task_id > 0) {

        MPI_Recv(&offset, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&loop_row, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&A, loop_row * nk, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&B, nk * nj, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);

        for (k = 0; k < nj; ++k) {
            for (i = 0; i < loop_row; ++i) {
                C[i][k] *= beta;
                for (j = 0; j < nk; ++j) {

                    C[i][k] += alpha * A[i][j] * B[j][k];

                }
            }
        }


        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&loop_row, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&C, loop_row * nj, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    bench_timer_stop();
    
    if (task_id) {
        exit(0);
    } else {

        avg_exec_time += bench_timer_print();
        printf("Average time spent with 0 threads on %d iterations: %lf\n", N, avg_exec_time);
        avg_exec_time = 0.0;
        //print_array(ni, nj, C);

    }
    return 0;
}