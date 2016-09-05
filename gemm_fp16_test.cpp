#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include "caffe/util/float16.hpp"
#include "caffe/util/math_functions.hpp"
using namespace std;
using namespace caffe;

void init_mat(float16 *A, const int M, const int N) {
    for (unsigned int i = 0; i < M; ++i) {
         for (unsigned int j = 0; j < N; ++j) {
             A[i*N+j] = rand() % 1000;
	 }
    }
}

void usage(string prog) {
    printf("Usage: %s SIZE", prog.c_str());
    exit(1);
}

int main(int argc, char* argv[]) {
    if (argc != 2) usage(argv[0]);
    const int SIZE = atoi(argv[1]);

    const int M = SIZE;
    const int N = SIZE;
    const int K = SIZE;

    float16 *h_A = (float16*)malloc(M*K*sizeof(float16));
    float16 *h_B = (float16*)malloc(K*N*sizeof(float16));
    float16 *h_C = (float16*)malloc(M*N*sizeof(float16));

    init_mat(h_A,M,K);
    init_mat(h_B,K,N);
 
    float16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float16));
    cudaMalloc(&d_B, K*N*sizeof(float16));
    cudaMalloc(&d_C, M*N*sizeof(float16));

    cudaMemcpy(d_A,h_A,M*K*sizeof(float16),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,K*N*sizeof(float16),cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    struct timespec start, end;
    long elapsed;

    cublasOperation_t cuTransA = CUBLAS_OP_N;
    cublasOperation_t cuTransB = CUBLAS_OP_T;
    const int lda = (cuTransA == CUBLAS_OP_N) ? K : M;
    const int ldb = (cuTransB == CUBLAS_OP_N) ? N : K;

    const float alpha_fp32 = (float)1.;
    const float beta_fp32 = (float)0.;
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &start);
    CUBLAS_CHECK(cublasSgemmEx(handle, cuTransB, cuTransA,
			    N, M, K, &alpha_fp32, d_B, CAFFE_DATA_HALF, ldb, d_A, CAFFE_DATA_HALF,
			    lda, &beta_fp32, d_C, CAFFE_DATA_HALF, N));
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed = (end.tv_sec*1e9+end.tv_nsec)-(start.tv_sec*1e9+start.tv_nsec);
    printf("%-18s %10ld ns\n", "<float16,float>", elapsed);


    const float16 alpha_fp16 = (float16)1.;
    const float16 beta_fp16 = (float16)0.;
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &start);
    CUBLAS_CHECK(cublasHgemm(handle, cuTransB, cuTransA,
			    N, M, K, &alpha_fp16.data, &d_B->data, ldb, &d_A->data,
			    lda, &beta_fp16.data, &d_C->data, N));
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed = (end.tv_sec*1e9+end.tv_nsec)-(start.tv_sec*1e9+start.tv_nsec);
    printf("%-18s %10ld ns\n", "<float16,float16>", elapsed);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}
