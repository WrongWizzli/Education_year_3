/* Include benchmark-specific header. */
#define MINI_DATASET
#include "gemm.h"
#include <omp.h>
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

void bench_timer_print() {
    printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
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

static void kernel_gemm_parallel2(int ni, int nj, int nk, double alpha, double beta, double C[ni][nj], double A[ni][nk], double B[nk][nj], int num_threads) {
    return;
}

int main(int argc, char** argv)
{
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int num_threads;
    double alpha;
    double beta;
    double (*C)[ni][nj]; C = (double(*)[ni][nj])malloc ((ni) * (nj) * sizeof(double));
    double (*A)[ni][nk]; A = (double(*)[ni][nk])malloc ((ni) * (nk) * sizeof(double));
    double (*B)[nk][nj]; B = (double(*)[nk][nj])malloc ((nk) * (nj) * sizeof(double));
    double (*D)[ni][nj]; D = (double(*)[ni][nj])malloc ((ni) * (nj) * sizeof(double));
    init_array (ni, nj, nk, &alpha, &beta, *C, *A, *B);
    if (argc >= 2 && !strcmp(argv[1], "1")) {

        printf("Enter the number of threads (maximum available is %d):\n", omp_get_max_threads());
        scanf("%d", &num_threads);

        bench_timer_start();

        kernel_gemm_parallel1(ni, nj, nk, alpha, beta, *C, *A, *B, num_threads);

        bench_timer_stop();
        bench_timer_print();
    } else if (argc >= 2 && !strcmp(argv[1], "2")) {

        printf("Enter the number of threads (maximum available is %d):\n", omp_get_max_threads());
        scanf("%d", &num_threads);

        bench_timer_start();

        kernel_gemm_parallel2(ni, nj, nk, alpha, beta, *C, *A, *B, num_threads);

        bench_timer_stop();
        bench_timer_print();
    } else {

        bench_timer_start();

        kernel_gemm(ni, nj, nk, alpha, beta, *C, *A, *B);

        bench_timer_stop();
        bench_timer_print();
    }
    if (argc >= 3 && !strcmp(argv[2], "print")) {
        print_array(ni, nj, *C);
    }
    free((void*)D);
    free((void*)C);
    free((void*)A);
    free((void*)B);

    return 0;
}
