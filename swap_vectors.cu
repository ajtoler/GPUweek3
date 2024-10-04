#include <stdio.h>


const int DSIZE = 40960;
const int block_size = 256;
const int grid_size = DSIZE/block_size;


__global__ void vector_swap(float *A, float *B, int v_size) {

    // Express the vector index in terms of threads and blocks
    int idx =  threadIdx.x + blockDim.x * blockIdx.x;

    // Swap the vector elements - make sure you are not out of range
    float temp = 0;
    if (idx < v_size) {
        temp = A[idx];
        A[idx] = B[idx];
        B[idx] = temp;
    }
}


int main() {

    float *h_A, *h_B, *d_A, *d_B;
    h_A = new float[DSIZE];
    h_B = new float[DSIZE];

    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate memory for host and device pointers 
    cudaMalloc(&d_A, DSIZE*sizeof(float));
    cudaMalloc(&d_B, DSIZE*sizeof(float));

    // Copy from host to device
    cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    printf("Pre Swap:\n");
    for (int i = 0; i < 5; i++) 
        printf("A[%d]: %f, B[%d]: %f\n", i, h_A[i], i, h_B[i]);
    vector_swap<<<grid_size, block_size>>>(d_A, d_B, DSIZE);

    // Copy back to host 
    cudaMemcpy(h_A, d_A, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

    // Print and check some elements to make sure swapping was successfull
    printf("Post Swap:\n");
    for (int i = 0; i < 5; i++) printf("A[%d]: %f, B[%d]: %f\n", i, h_A[i], i, h_B[i]);

    // Free the memory 
    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
