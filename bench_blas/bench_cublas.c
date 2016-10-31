// based on NVIDIA's example http://docs.nvidia.com/cuda/cublas/#axzz4NESjLpCM
//Example 2. Application Using C and CUBLAS: 0-based indexing
//-----------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <strings.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
// ld: matrix rows (fortran-order)
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
    cublasSscal (handle, n-p, &alpha, &m[IDX2C(p,q,ldm)], ldm);
    cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
}

static int run_sgemm_times(int times,
cublasHandle_t handle,
cublasOperation_t transa, cublasOperation_t transb,
int m, int n, int k,
const float          *alpha,
const float          *A, int lda,
const float          *B, int ldb,
const float          *beta,
float          *C, int ldc) {
    for (int i = 0; i < times; i++) {
        cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

static void fill_mat(float* m, int rows, int cols) {
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            m[IDX2C(i,j,rows)] = (float)(i + j * rows + 1);
        }
    }
}

static void do_cudaMalloc(void** devPtr, int size) {
    cudaError_t cudaStat = cudaMalloc (devPtr, size);
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        exit(EXIT_FAILURE);
    }
}

int is_trans(const char* c) {
    if (strcasecmp(c, "N") == 0) {
        return 0;
    } else if (strcasecmp(c, "T") == 0) {
        return 1;
    } else {
        printf("Unknown transposition directive\n");
        exit(EXIT_FAILURE);
    }
}

int main (int argc, char *argv[]){
    cudaError_t cudaStat;    
    cublasStatus_t stat;
    cublasHandle_t handle;
    int M, N, K;// M*K * K*N = M*N
    M = atol(argv[1]); N = atol(argv[2]); K = atol(argv[3]);
    int transA = is_trans(argv[4]);
    int transB = is_trans(argv[5]);
    int exec_times_offset = 10;
    int exec_times_actual = 20;
    int i, j;
    float *devPtrA, *devPtrB, *devPtrC;
    float *a, *b, *c;
    a = (float *)malloc (M * K * sizeof (*a));
    b = (float *)malloc (K * N * sizeof (*a));
    c = (float *)malloc (M * N * sizeof (*a));
    if (!a || !b || !c) {
        printf ("host memory allocation failed");
        exit(EXIT_FAILURE);
    }
    fill_mat(a, M, K);
    fill_mat(b, K, N);
    fill_mat(c, M, N);
    do_cudaMalloc((void**)&devPtrA, M*K*sizeof(*a));
    do_cudaMalloc((void**)&devPtrB, K*N*sizeof(*b));
    do_cudaMalloc((void**)&devPtrC, M*N*sizeof(*c));
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    long long exec_us[3];
    for (int run_i = 0; run_i < 3; run_i++){
        struct timeval tv_begin, tv_end;
        gettimeofday(&tv_begin, NULL);
    stat = cublasSetMatrix (M, K, sizeof(*a), a, M, devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }
    stat = cublasSetMatrix (K, N, sizeof(*b), b, K, devPtrB, K);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrB);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }
    float alpha = 1.0F, beta = 0.0F;
    run_sgemm_times(exec_times_offset + exec_times_actual * run_i, handle, transA ? CUBLAS_OP_T : CUBLAS_OP_N, transB ? CUBLAS_OP_T : CUBLAS_OP_N, M, N, K,
    &alpha,
    devPtrA, transA ? K : M,
    devPtrB, transB ? N : K,
    &beta,
    devPtrC, M);
    stat = cublasGetMatrix (M, N, sizeof(*c), devPtrC, M, c, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrC);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }
        gettimeofday(&tv_end, NULL);
        long long elapsed_usec = ((long long)tv_end.tv_sec * 1000000LL + (long long)tv_end.tv_usec) - ((long long)tv_begin.tv_sec * 1000000LL + (long long)tv_begin.tv_usec);
        exec_us[run_i] = elapsed_usec;
    }
    //printf("%lld %lld %lld\n", exec_us[0], exec_us[1], exec_us[2]);
    double time_per_one_calc_ms = (double)(exec_us[2] - exec_us[1]) / exec_times_actual / 1000;
    printf("%d,%d,%d,%d,%d,%f\n", M, N, K, transA, transB, time_per_one_calc_ms);
    cudaFree (devPtrA);
    cudaFree (devPtrB);
    cudaFree (devPtrC);
    cublasDestroy(handle);
    free(a);
    free(b);
    free(c);
    return EXIT_SUCCESS;
}
