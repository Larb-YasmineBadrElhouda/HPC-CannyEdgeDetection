#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<raylib.h>

#define SCREEN_W 640
#define SCREEN_H 480 

// threshold values
#define UPPER_T 40 
#define LOWER_T 10 

typedef enum{
  HORIZONTAL,
  VERTICAL,
  L_DIAGONAL,
  R_DIAGONAL
}GradDir;

typedef enum{
  SUPPRESSED,
  WEAK,
  STRONG
}EdgePower;

typedef struct {
  GradDir dir;
  EdgePower power;
}PixelGrad;

Color *toGrayScale(Color *src_pixels, int img_w, int img_h){

  Color *gray_pixels = calloc(img_w * img_h, sizeof(Color));

  for(int i = 0; i < img_h; ++i){
    for(int j = 0; j < img_w; ++j){
      Color pixel = src_pixels[img_w * i + j];

      unsigned char lum = (unsigned char) 0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b;

      Color gray_pixel = {.r = lum, .g = lum, .b = lum, .a = pixel.a};

      gray_pixels[img_w * i + j] = gray_pixel;
    }
  }
  return gray_pixels;
}

Color *gaussianBlur(Color *pixels,size_t img_w, size_t img_h){
  float filter[3][3] = {{0.07511361,0.1238414,0.07511361},{0.1238414,0.20417996 ,0.1238414 }, {0.07511361, 0.1238414,  0.07511361}};
  Color *blured= calloc(img_w * img_h, sizeof(Color));

  for (size_t i = 0; i < img_h; ++i)
  {
    for (size_t j = 0; j < img_w; ++j)
    {
      float val = 0;
      // loop through the nine neighbouring pixels
      for (int k = -1; k <= 1; ++k)
      {
        for (int l = -1; l <= 1; ++l)
        {
          if (i + k < 0 || i + k >= img_h)
            continue;
          if (j + l < 0 || j + l >= img_w)
            continue;
          val += (pixels[img_w * (i + k) + (j + l)].r) * filter[1 + k][1 + l];
        }
      }
      blured[img_w * i + j].r = (unsigned char)fmin(fmax(val, 0), 255);
      blured[img_w * i + j].g = blured[img_w * i + j].r;
      blured[img_w * i + j].b = blured[img_w * i + j].r;
      blured[img_w * i + j].a = pixels[img_w * i + j].a; // Preserve the original alpha
    }
  }

  return blured;

}

Color *detectEdge(Color *pixels, PixelGrad *grad_dir, int img_w, int img_h)
{
  Color *y = calloc(img_w * img_h, sizeof(Color));

   float gx[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}}; // sobel operator
   float gy[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

  for (int i = 0; i < img_h; ++i)
  {
    for (int j = 0; j < img_w; ++j)
    {
      float xgrad = 0;
      float ygrad = 0;
      // loop through the nine neighbouring pixels
      for (int k = -1; k <= 1; ++k)
      {
        for (int l = -1; l <= 1; ++l)
        {
          if (i + k < 0 || i + k >= img_h)
            continue;
          if (j + l < 0 || j + l >= img_w)
            continue;
          xgrad += (pixels[img_w * (i + k) + (j + l)].r) * gx[1 + k][1 + l];
          ygrad += (pixels[img_w * (i + k) + (j + l)].r) * gy[1 + k][1 + l];
        }
      }
      // Gradient direction
      float theta = atan2f(ygrad,xgrad);
      
      if( theta > 22.5 && theta <= 67.5)
        grad_dir[img_w * i + j].dir = R_DIAGONAL; 

      else if(theta > 67.5 && theta <= 112.5) 
        grad_dir[img_w * i + j].dir = VERTICAL; 

      else if( theta > 112.5 && theta <= 157.5)
        grad_dir[img_w * i + j].dir = L_DIAGONAL; 
      else
        grad_dir[img_w * i + j].dir = HORIZONTAL; 


      double mag = sqrt(pow(xgrad,2) + pow(ygrad,2));

      y[img_w * i + j].r = (unsigned char)fmin(fmax(mag, 0), 255);
      y[img_w * i + j].g = y[img_w * i + j].r;
      y[img_w * i + j].b = y[img_w * i + j].r;
      y[img_w * i + j].a = pixels[img_w * i + j].a; // Preserve the original alpha

      if(mag >= UPPER_T)
        grad_dir[img_w * i + j].power = STRONG;
      else if(mag >= LOWER_T && mag < UPPER_T)
        grad_dir[img_w * i + j].power = WEAK;
      else
        grad_dir[img_w * i + j].power = SUPPRESSED;

    }
  }
  return y;
}

void setPixelValue(Color *pixels, int idx, int val){
  pixels[idx].r = val;
  pixels[idx].g = val;
  pixels[idx].b = val;
}

void nonMaximumSuppression(Color *pixels,PixelGrad *dir, int img_w, int img_h){

  for(int i = 0; i < img_h; ++i){
    for(int j = 0; j < img_w; ++j){
      // loop through all the eight neighbours of the particular pixel
      // if power suppressed then continues

        for(int k = -1; k <= 1; ++k){
          for(int l = -1; l <= 1; ++l){
            if(i + k < 0 || i + k >= img_h)
              continue;
            if(j + l < 0 || j + l >= img_w)
              continue;
            if(k == 0 && l == 0)
              continue;
            if(dir[img_w * i + j].dir == dir[img_w * (i + k) + (j + l)].dir){

              if(pixels[img_w * i + j].r < pixels[img_w * (i + k) + (j + l)].r){
                setPixelValue(pixels, img_w * i + j, 0);
                dir[img_w * i + j].power = SUPPRESSED;
                break;
              }
          }
        }
      }
    }
  }
}


void edgeTrackingHysteresis(Color *pixels, PixelGrad *dir,int img_w, int img_h){
  
  for(int i = 0; i < img_h; ++i){
    for(int j = 0; j < img_w; ++j){
      if(dir[img_w * i + j].power == STRONG){
        setPixelValue(pixels, img_w * i + j, 255);
        continue;
      }
      if(dir[img_w * i + j].power == SUPPRESSED){
        setPixelValue(pixels, img_w * i + j, 0);
        continue;
      }
      // if the edge is WEAK
      for(int k = -1; k <= 1; ++k){
        if(dir[img_w * i + j].power == STRONG)
          break;

        for(int l = -1; l <= 1; ++l){
          if(dir[img_w * (i + k) + (j + l)].power == STRONG){
            dir[img_w * i + j].power = STRONG;
            setPixelValue(pixels, img_w * i + j, 255);
          }
        }

        dir[img_w * i + j].power = SUPPRESSED;
        setPixelValue(pixels, img_w * i + j, 0);
      }
    }
  }
}


int main(int argc, char **argv){

  if(argc < 2){
    perror("Usage `./sobel filename.png");
    return 1;
  }
  
  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(SCREEN_W, SCREEN_H, "Canny");

  // Read Image
  Image src_image = LoadImage(argv[1]);
  int img_w = src_image.width;
  int img_h = src_image.height;
  // get input pixel data
  Color *pixels = LoadImageColors(src_image);
  // convert to gray scale
  Color *gray_pixels = toGrayScale(pixels,img_w, img_h);
  // Gaussian blue
  Color *blured_pixels= gaussianBlur(gray_pixels,img_w, img_h);
  // Sobel edge
  PixelGrad *grad_dir = calloc(img_w * img_h, sizeof(PixelGrad));
  Color *sobel_pixels= detectEdge(blured_pixels,grad_dir,img_w, img_h);
  nonMaximumSuppression(sobel_pixels,grad_dir,img_w,img_h);
  edgeTrackingHysteresis(sobel_pixels,grad_dir,img_w,img_h);
  // Load texture from iamge
  Image gray_image = {.data = sobel_pixels,
    .width = img_w,
    .height = img_h,
    .mipmaps = 1,
    .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,      // 32 bpp
  };
  //Texture2D texture = LoadTextureFromImage(src_image);
  Texture2D texture = LoadTextureFromImage(gray_image);

  while(!WindowShouldClose()){
    BeginDrawing();
      ClearBackground(WHITE);  // Set background color (framebuffer clear color)
      DrawTexture(texture,SCREEN_W/2 - img_w / 2, SCREEN_H/2 - img_h /2, WHITE);
    EndDrawing();
  }
  UnloadTexture(texture);
  UnloadImageColors(pixels);
  free(gray_pixels);
  free(blured_pixels);
  free(sobel_pixels);
  CloseWindow();
  return 0;
}