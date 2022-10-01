#ifndef _GEMM_H
#define _GEMM_H 


# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#define SMALL_DATASET
# endif

# if !defined(NI) && !defined(NJ) && !defined(NK)

# ifdef MINI_DATASET
#define NI 20
#define NJ 25
#define NK 30
# endif

# ifdef SMALL_DATASET
#define NI 60
#define NJ 70
#define NK 80
# endif

# ifdef MEDIUM_DATASET
#define NI 200
#define NJ 220
#define NK 240
# endif

# ifdef LARGE_DATASET
#define NI 1000
#define NJ 1100
#define NK 1200
# endif

# ifdef EXTRALARGE_DATASET
#define NI 2000
#define NJ 2300
#define NK 2600
# endif

#endif

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#endif
