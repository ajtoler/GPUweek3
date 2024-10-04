#include <stdio.h>

const int DSIZE_X = 256;
const int DSIZE_Y = 256;

__global__ void add_matrix(const float *A, const float *B, float *C, const int DSIZE_X, const int DIZE_Y)
{
    // Express in terms of threads and blocks
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    // Add the two matrices - make sure you are not out of range
    if (idx <  DSIZE_X && idy < DIZE_Y)
        C[idy * DSIZE_Y + idx] =  A[idy * DSIZE_Y + idx] + B[idy * DSIZE_Y + idx];

}

int main()
{

    // Create and allocate memory for host and device pointers 
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    int DSIZE = DSIZE_X * DSIZE_Y;

    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];

    cudaMalloc(&d_A, DSIZE*sizeof(float));
    cudaMalloc(&d_B, DSIZE*sizeof(float));
    cudaMalloc(&d_C, DSIZE*sizeof(float));
    
    // Fill in the matrices
    for (int i = 0; i < DSIZE_X; i++) {
        for (int j = 0; j < DSIZE_Y; j++) {
            h_A[i * DSIZE_Y + j] = rand()/(float)RAND_MAX;
            h_B[i * DSIZE_Y + j] = rand()/(float)RAND_MAX;
            h_C[i * DSIZE_Y + j] = 0;
        }
    }

    // Copy from host to device
    cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    // dim3 is a built in CUDA type that allows you to define the block 
    // size and grid size in more than 1 dimentions
    // Syntax : dim3(Nx,Ny,Nz)
    dim3 blockSize(16, 16); 
    dim3 gridSize((DSIZE_X + blockSize.x - 1) / blockSize.x, (DSIZE_Y + blockSize.y - 1) / blockSize.y); 
    
    add_matrix<<<gridSize, blockSize>>>(d_A, d_B, d_C, DSIZE_X, DSIZE_Y);

    // Copy back to host 
    cudaMemcpy(h_C, d_C, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

    // Print and check some elements to make the addition was succesfull
    printf("A:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%f ", h_A[i * DSIZE_Y + j]);
        }
        printf("\n");
    }

    printf("B:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%f ", h_B[i * DSIZE_Y + j]);
        }
        printf("\n");
    }

    printf("C:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%f ", h_C[i * DSIZE_Y + j]);
        }
        printf("\n");
    }

    // Free the memory    
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B); 
    cudaFree(d_C);

    return 0;
}