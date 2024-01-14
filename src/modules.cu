#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>

#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

#define N_GLOB  1024

__global__ void vectorAdd(float* d_A, float* d_B, float* d_c){ 
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id < N_GLOB) {
        d_c[id] = d_A[id] + d_B[id];
    }

}

__global__ void matmulKernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < K) {
        float sum = 0.0;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

void checkCudaError(cudaError_t error) {
    if(error != cudaSuccess) {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}



int main (void){
    int M = 5;
    int N = 3;
    int K = 4;

    float* h_A, *h_B, *h_c;
    h_A = (float *)malloc(M*N*sizeof(float));
    h_B = (float *)malloc(N*K*sizeof(float));
    h_c  = (float *)malloc(M*K*sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_A[j + N*i] = j + N*i + 3;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            h_B[j + K*i] = j + M*i + 1;
        }
    }
    
    
    float *d_A, *d_B, *d_c;

    checkCudaError(cudaMalloc((void **)&d_A, M * N * sizeof(float)));
    checkCudaError(cudaMalloc((void **)&d_B, N * K * sizeof(float)));
    checkCudaError(cudaMalloc((void **)&d_c, M * K * sizeof(float)));

    // Copy matrices to the GPU
    checkCudaError(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice));

    // Define grid and block sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    matmulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_c, M, N, K);
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    // Copy the result back to the host
    checkCudaError(cudaMemcpy(h_c, d_c, M * K * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the result
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%f ", h_c[j + K * i]);
        }
        printf("\n");
    }

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_c);
        return EXIT_SUCCESS;


}