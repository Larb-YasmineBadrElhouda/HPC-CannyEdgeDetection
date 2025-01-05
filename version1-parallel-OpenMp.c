#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <raylib.h>
#include <omp.h> // OpenMP

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

// Convert to grayscale (parallelized)
Color *toGrayScale(Color *src_pixels, int img_w, int img_h) {
    Color *gray_pixels = calloc(img_w * img_h, sizeof(Color));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < img_h; ++i) {
        for (int j = 0; j < img_w; ++j) {
            Color pixel = src_pixels[img_w * i + j];
            unsigned char lum = (unsigned char)(0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b);
            gray_pixels[img_w * i + j] = (Color){.r = lum, .g = lum, .b = lum, .a = pixel.a};
        }
    }
    return gray_pixels;
}

// Apply Gaussian Blur (parallelized)
Color *gaussianBlur(Color *pixels, size_t img_w, size_t img_h) {
    float filter[3][3] = {
        {0.07511361, 0.1238414, 0.07511361},
        {0.1238414,  0.20417996, 0.1238414},
        {0.07511361, 0.1238414,  0.07511361}
    };
    Color *blured = calloc(img_w * img_h, sizeof(Color));
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < img_h; ++i) {
        for (size_t j = 0; j < img_w; ++j) {
            float val = 0;
            for (int k = -1; k <= 1; ++k) {
                for (int l = -1; l <= 1; ++l) {
                    int ni = i + k, nj = j + l;
                    if (ni >= 0 && ni < img_h && nj >= 0 && nj < img_w) {
                        val += pixels[img_w * ni + nj].r * filter[k + 1][l + 1];
                    }
                }
            }
            unsigned char blur_val = (unsigned char)fmin(fmax(val, 0), 255);
            blured[img_w * i + j] = (Color){.r = blur_val, .g = blur_val, .b = blur_val, .a = pixels[img_w * i + j].a};
        }
    }
    return blured;
}

// Sobel Edge Detection (parallelized)
Color *detectEdge(Color *pixels, PixelGrad *grad_dir, int img_w, int img_h) {
    float gx[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    float gy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
    Color *y = calloc(img_w * img_h, sizeof(Color));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < img_h; ++i) {
        for (int j = 0; j < img_w; ++j) {
            float xgrad = 0, ygrad = 0;
            for (int k = -1; k <= 1; ++k) {
                for (int l = -1; l <= 1; ++l) {
                    int ni = i + k, nj = j + l;
                    if (ni >= 0 && ni < img_h && nj >= 0 && nj < img_w) {
                        xgrad += pixels[img_w * ni + nj].r * gx[k + 1][l + 1];
                        ygrad += pixels[img_w * ni + nj].r * gy[k + 1][l + 1];
                    }
                }
            }
            float theta = atan2f(ygrad, xgrad);
            grad_dir[img_w * i + j].dir = (theta > 22.5 && theta <= 67.5) ? R_DIAGONAL
                                          : (theta > 67.5 && theta <= 112.5) ? VERTICAL
                                          : (theta > 112.5 && theta <= 157.5) ? L_DIAGONAL
                                          : HORIZONTAL;

            float mag = sqrt(xgrad * xgrad + ygrad * ygrad);
            unsigned char edge_val = (unsigned char)fmin(fmax(mag, 0), 255);
            y[img_w * i + j] = (Color){.r = edge_val, .g = edge_val, .b = edge_val, .a = pixels[img_w * i + j].a};

            grad_dir[img_w * i + j].power = (mag >= UPPER_T) ? STRONG
                                         : (mag >= LOWER_T) ? WEAK
                                         : SUPPRESSED;
        }
    }
    return y;
}

int main() {
    InitWindow(SCREEN_W, SCREEN_H, "Edge Detection with OpenMP");
    Image img = LoadImage("your_image_path.png");
    ImageResize(&img, SCREEN_W, SCREEN_H);

    Color *pixels = LoadImageColors(img);
    Color *grayscale = toGrayScale(pixels, img.width, img.height);
    Color *blurred = gaussianBlur(grayscale, img.width, img.height);

    PixelGrad *grad_dir = calloc(img.width * img.height, sizeof(PixelGrad));
    Color *edges = detectEdge(blurred, grad_dir, img.width, img.height);

    Image edge_img = GenImageColor(img.width, img.height, BLACK);
    SetImageColor(edge_img, edges);
    Texture2D edge_texture = LoadTextureFromImage(edge_img);

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(BLACK);
        DrawTexture(edge_texture, 0, 0, WHITE);
        EndDrawing();
    }

    UnloadTexture(edge_texture);
    UnloadImage(edge_img);
    UnloadImage(img);
    free(pixels);
    free(grayscale);
    free(blurred);
    free(edges);
    free(grad_dir);
    CloseWindow();

    return 0;
}
