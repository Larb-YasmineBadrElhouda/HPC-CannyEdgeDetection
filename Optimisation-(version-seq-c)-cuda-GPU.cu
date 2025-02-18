#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <raylib.h>

#define SCREEN_W 640
#define SCREEN_H 480
#define UPPER_T 40
#define LOWER_T 10

typedef enum {
    HORIZONTAL,
    VERTICAL,
    L_DIAGONAL,
    R_DIAGONAL
} GradDir;

typedef enum {
    SUPPRESSED,
    WEAK,
    STRONG
} EdgePower;

typedef struct {
    GradDir dir;
    EdgePower power;
} PixelGrad;

// Optimized Kernel for Grayscale
__global__ void toGrayScaleKernel(Color *src_pixels, unsigned char *gray_pixels, int img_w, int img_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * img_w + x;

    if (x < img_w && y < img_h) {
        Color pixel = src_pixels[idx];
        gray_pixels[idx] = (unsigned char)(0.299f * pixel.r + 0.587f * pixel.g + 0.114f * pixel.b);
    }
}

// Optimized Kernel for Gaussian Blur with Shared Memory
__global__ void gaussianBlurKernelShared(unsigned char *gray_pixels, unsigned char *blured_pixels, int img_w, int img_h) {
    __shared__ unsigned char shared_block[18][18];  // Shared memory block for 16x16 + halo
    const float filter[3][3] = {
        {0.07511361, 0.1238414, 0.07511361},
        {0.1238414,  0.20417996, 0.1238414},
        {0.07511361, 0.1238414,  0.07511361}
    };

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + 1;  // Local thread indices for shared memory
    int ty = threadIdx.y + 1;

    // Load data into shared memory
    if (x < img_w && y < img_h) {
        shared_block[ty][tx] = gray_pixels[y * img_w + x];
        if (threadIdx.x == 0 && x > 0) shared_block[ty][tx - 1] = gray_pixels[y * img_w + x - 1]; // Left halo
        if (threadIdx.y == 0 && y > 0) shared_block[ty - 1][tx] = gray_pixels[(y - 1) * img_w + x]; // Top halo
        if (threadIdx.x == blockDim.x - 1 && x < img_w - 1) shared_block[ty][tx + 1] = gray_pixels[y * img_w + x + 1]; // Right halo
        if (threadIdx.y == blockDim.y - 1 && y < img_h - 1) shared_block[ty + 1][tx] = gray_pixels[(y + 1) * img_w + x]; // Bottom halo
    }
    __syncthreads();

    // Apply Gaussian filter
    if (x < img_w && y < img_h) {
        float sum = 0.0f;
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                sum += shared_block[ty + ky][tx + kx] * filter[ky + 1][kx + 1];
            }
        }
        blured_pixels[y * img_w + x] = (unsigned char)fminf(fmaxf(sum, 0.0f), 255.0f);
    }
}

// Optimized Kernel for Sobel Edge Detection with Gradient Direction
__global__ void sobelEdgeKernelShared(unsigned char *blured_pixels, unsigned char *edges, PixelGrad *grad_dir, int img_w, int img_h) {
    __shared__ unsigned char shared_block[18][18];  // Shared memory block for 16x16 + halo
    const int gx[3][3] = { {1, 0, -1}, {2, 0, -2}, {1, 0, -1} };
    const int gy[3][3] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + 1;  // Local thread indices for shared memory
    int ty = threadIdx.y + 1;

    // Load data into shared memory
    if (x < img_w && y < img_h) {
        shared_block[ty][tx] = blured_pixels[y * img_w + x];
        if (threadIdx.x == 0 && x > 0) shared_block[ty][tx - 1] = blured_pixels[y * img_w + x - 1]; // Left halo
        if (threadIdx.y == 0 && y > 0) shared_block[ty - 1][tx] = blured_pixels[(y - 1) * img_w + x]; // Top halo
        if (threadIdx.x == blockDim.x - 1 && x < img_w - 1) shared_block[ty][tx + 1] = blured_pixels[y * img_w + x + 1]; // Right halo
        if (threadIdx.y == blockDim.y - 1 && y < img_h - 1) shared_block[ty + 1][tx] = blured_pixels[(y + 1) * img_w + x]; // Bottom halo
    }
    __syncthreads();

    // Compute gradients and magnitude
    if (x < img_w && y < img_h) {
        float gx_sum = 0.0f, gy_sum = 0.0f;
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                gx_sum += shared_block[ty + ky][tx + kx] * gx[ky + 1][kx + 1];
                gy_sum += shared_block[ty + ky][tx + kx] * gy[ky + 1][kx + 1];
            }
        }
        float mag = sqrtf(gx_sum * gx_sum + gy_sum * gy_sum);
        float angle = atan2f(gy_sum, gx_sum);

        // Assign gradient direction
        grad_dir[y * img_w + x].dir = (angle > 22.5f && angle <= 67.5f) ? R_DIAGONAL :
                                      (angle > 67.5f && angle <= 112.5f) ? VERTICAL :
                                      (angle > 112.5f && angle <= 157.5f) ? L_DIAGONAL : HORIZONTAL;

        // Edge classification
        edges[y * img_w + x] = (unsigned char)fminf(fmaxf(mag, 0.0f), 255.0f);
        grad_dir[y * img_w + x].power = (mag >= UPPER_T) ? STRONG : (mag >= LOWER_T) ? WEAK : SUPPRESSED;
    }
}

int main() {
   // the main -----------------------
    if(argc < 2){
    perror("Usage `./sobel filename.png");
    return 1;
	/! ------------------
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	/!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  }
}
