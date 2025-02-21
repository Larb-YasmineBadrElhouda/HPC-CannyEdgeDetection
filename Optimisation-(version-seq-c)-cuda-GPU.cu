#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <raylib.h>

#define UPPER_T 40  // le seuil haut
#define LOWER_T 10  //  bas

// direction du gradient
typedef enum {
    HORIZONTAL,
    VERTICAL,
    L_DIAGONAL,
    R_DIAGONAL
} GradDir;

// forces du gradient
typedef enum {
    SUPPRESSED,
    WEAK,
    STRONG
} EdgePower;

// ou on stocke le gradient 
typedef struct {
    GradDir dir;
    EdgePower power;
} PixelGrad;

// ===============================
// KERNEL : Convertir en niveaux de gris
// ===============================
__global__ void toGrayScaleKernel(Color *src_pixels, unsigned char *gray_pixels, int img_w, int img_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * img_w + x;

    if (x < img_w && y < img_h) {
        Color pixel = src_pixels[idx];
        gray_pixels[idx] = (unsigned char)(0.299f * pixel.r + 0.587f * pixel.g + 0.114f * pixel.b);
    }
}

// ===============================
// KERNEL : Détection de contours avec Sobel
// ===============================
__global__ void sobelEdgeKernel(unsigned char *gray_pixels, unsigned char *edges, PixelGrad *grad_dir, int img_w, int img_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * img_w + x;

    if (x >= img_w || y >= img_h) return;

    // Opérateurs Sobel
    const int gx[3][3] = { {1, 0, -1}, {2, 0, -2}, {1, 0, -1} };
    const int gy[3][3] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };

    float gx_sum = 0.0f, gy_sum = 0.0f;
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int neighbor_idx = (y + ky) * img_w + (x + kx);
            if (neighbor_idx >= 0 && neighbor_idx < img_w * img_h) {
                gx_sum += gray_pixels[neighbor_idx] * gx[ky + 1][kx + 1];
                gy_sum += gray_pixels[neighbor_idx] * gy[ky + 1][kx + 1];
            }
        }
    }

    float mag = sqrtf(gx_sum * gx_sum + gy_sum * gy_sum);
    grad_dir[idx].power = (mag >= UPPER_T) ? STRONG : (mag >= LOWER_T) ? WEAK : SUPPRESSED;
    edges[idx] = (unsigned char)fminf(fmaxf(mag, 0.0f), 255.0f);
}

// ===============================
// MAIN
// ===============================
int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: ./sobel filename.png\n");
        return 1;
    }

    // Chargmnt  d 'image
    Image src_image = LoadImage(argv[1]);
    int img_w = src_image.width;
    int img_h = src_image.height;
    Color *pixels = LoadImageColors(src_image);

    // Alloc mémoire :  GPU
    Color *d_pixels;
    unsigned char *d_gray, *d_edges;
    PixelGrad *d_grad_dir;
    
    cudaMalloc((void**)&d_pixels, img_w * img_h * sizeof(Color));
    cudaMalloc((void**)&d_gray, img_w * img_h * sizeof(unsigned char));
    cudaMalloc((void**)&d_edges, img_w * img_h * sizeof(unsigned char));
    cudaMalloc((void**)&d_grad_dir, img_w * img_h * sizeof(PixelGrad));

    // Copy image sur le GPU
    cudaMemcpy(d_pixels, pixels, img_w * img_h * sizeof(Color), cudaMemcpyHostToDevice);

    //  taill  blocs et  la grille
    dim3 blockSize(16, 16);
    dim3 gridSize((img_w + 15) / 16, (img_h + 15) / 16);

    // Exéc kernel :  convertir en niveaux de gris
    toGrayScaleKernel<<<gridSize, blockSize>>>(d_pixels, d_gray, img_w, img_h);
    cudaDeviceSynchronize();

    // Exécuter kernel : Sobel
    sobelEdgeKernel<<<gridSize, blockSize>>>(d_gray, d_edges, d_grad_dir, img_w, img_h);
    cudaDeviceSynchronize();

    // Copy results  GPU ----> CPU
    unsigned char *h_edges = (unsigned char*)malloc(img_w * img_h * sizeof(unsigned char));
    cudaMemcpy(h_edges, d_edges, img_w * img_h * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Raylib : Affichage  results
    Image edgeImage = { .data = h_edges, .width = img_w, .height = img_h, .format = PIXELFORMAT_UNCOMPRESSED_GRAYSCALE, .mipmaps = 1 };
    Texture2D edgeTexture = LoadTextureFromImage(edgeImage);

    InitWindow(img_w, img_h, "CUDA Sobel Edge Detection");
    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(BLACK);
        DrawTexture(edgeTexture, 0, 0, WHITE);
        EndDrawing();
    }

    // Liber mémoire
    UnloadTexture(edgeTexture);
    UnloadImage(src_image);
    free(h_edges);
    cudaFree(d_pixels);
    cudaFree(d_gray);
    cudaFree(d_edges);
    cudaFree(d_grad_dir);

    CloseWindow();
    return 0;
}
