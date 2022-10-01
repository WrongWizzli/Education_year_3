#define EXTRALARGE_DATASET
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

static void kernel_gemm_parallel1(int ni, int nj, int nk, double alpha, double beta, double C[ni][nj], double A[ni][nk], double B[nk][nj], int num_threads) {
    int i, j, k;
    omp_set_num_threads(num_threads);
    #pragma omp parallel for private(j, k)
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
             C[i][j] *= beta;
            for (k = 0; k < nk; k++) {
                C[i][j] += alpha * A[i][k] * B[k][j];
            }
        }
    }
}

static void smart_transpose(int nk, int nj, double B[nk][nj], double D[nj][nk]) {

    int i, j, k, l;
    int block_size = 16;

    #pragma omp parallel for private(j, k, l)
    for (i = 0; i < nk; i += block_size) {
        for (j = 0; j < nj; j += block_size) {
            for (k = 0; k < block_size && i + k < nk; ++k) {
                for (l = 0; l < block_size && j + l < nj; ++l) {
                    D[j + l][i + k] = B[i + k][j + l];
                }
            }
        }
    }
}
static void kernel_gemm_parallel2(int ni, int nj, int nk, double alpha, double beta, double C[ni][nj], double A[ni][nk], double B[nk][nj], int num_threads) {

    omp_set_num_threads(num_threads);

    double (*D)[nj][nk]; D = (double(*)[nj][nk])malloc ((nj) * (nk) * sizeof(double));

    smart_transpose(nk, nj, B, *D);

    int i, j, k;

    #pragma omp parallel for private(j, k)
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
             C[i][j] *= beta;
            for (k = 0; k < nk; k++) {
                C[i][j] += alpha * A[i][k] * B[j][k];
            }
        }
    }
}
int NOT_SET_SIZE = -1;
static void kernel_gemm_parallel3(int ni, int nj, int nk, double alpha, double beta, double C[ni][nj], double A[ni][nk], double B[nk][nj], int num_threads, int block_size) {

    omp_set_num_threads(num_threads);

    if (block_size == NOT_SET_SIZE) {
        block_size = 16;
    }
    int i, j, k, i0, j0, k0;

    #pragma omp parallel for private(i, j)
    for (i = 0; i < ni; ++i) {
        for (j = 0; j < nj; ++j) {
            C[i][j] *= beta;
        }
    }

    for (i = 0; i < ni; i += block_size) {
        for (j = 0; j < nj; j += block_size) {
            for (k = 0; k < nk; k += block_size) {
            #pragma omp task private(i0, j0, k0) firstprivate(i, j, k) \
                    depend(in: A[i:block_size][k:block_size], B[k:block_size][j:block_size]) \
                    depend(inout: C[i:block_size][j:block_size])
                for (i0 = i; i0 < block_size + i && i0 < ni; ++i0) {
                    for (j0 = j; j0 < block_size + j && j0 < nj; ++j0) {
                        for (k0 = k; k0 < block_size + k && k0 < nk; ++k0) {
                            C[i0][j0] += alpha * A[i0][k0] * B[k0][j0];
                        }
                    }
                }

            }
        }
    }

}

int main(int argc, char** argv) {

    int ni = NI;
    int nj = NJ;
    int nk = NK;

    double alpha;
    double beta;
    double (*C)[ni][nj]; C = (double(*)[ni][nj])malloc ((ni) * (nj) * sizeof(double));
    double (*A)[ni][nk]; A = (double(*)[ni][nk])malloc ((ni) * (nk) * sizeof(double));
    double (*B)[nk][nj]; B = (double(*)[nk][nj])malloc ((nk) * (nj) * sizeof(double));

    int iter = 5, j;

    printf("Enter number of iterations:\n");
    scanf("%d", &iter);

    int NUM_THREADS[] = {2, 4, 8, 16, 32, 64, 128, 160};

    double avg_exec_time = 0.0;
    int num_threads = 4;

    for (j = 0; j < 8; j++) {

        printf("--------------------------------\n");
        printf("Number of threads: %d\nNumber of iterations: %d\n", NUM_THREADS[j], iter);

        int i;
        int num_threads = NUM_THREADS[j];

        for (i = 0; i < iter; i++) {

            init_array(ni, nj, nk, &alpha, &beta, *C, *A, *B);

            bench_timer_start();

            kernel_gemm_parallel1(ni, nj, nk, alpha, beta, *C, *A, *B, num_threads);

            bench_timer_stop();
            avg_exec_time += bench_timer_print();
        }

        printf("Average time spent with %d threads on %d iterations: %lf\n", num_threads, iter, avg_exec_time / iter);
        avg_exec_time = 0.0;

    }

    printf("------------------------------------\n");
    printf("Sequantial multiplication with %d iterations\n", iter);

        for (j = 0; j < iter; j++) {

            init_array(ni, nj, nk, &alpha, &beta, *C, *A, *B);

            bench_timer_start();

            kernel_gemm(ni, nj, nk, alpha, beta, *C, *A, *B);

            bench_timer_stop();
            avg_exec_time += bench_timer_print();
        }

        printf("Average time spent with sequantial method on %d iterations: %lf\n", iter, avg_exec_time / iter);
        avg_exec_time = 0.0;

    if (argc >= 2 && !strcmp(argv[1], "print")) {
        print_array(ni, nj, *C);
    }

    free((void*)C);
    free((void*)A);
    free((void*)B);

    return 0;
}

