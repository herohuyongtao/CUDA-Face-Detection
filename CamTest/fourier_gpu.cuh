//fourier_gpu.h
#pragma once
/*-------STRUCTURES---------*/
typedef struct {long rows; long cols; unsigned char* data;} sImageFourier;

/*-------FUNCTIONS----------*/
void FourierTransformGPU(BITMAPINFO *bmpInfo, void *bmpData, bool inverse, float img[PREF_WIDTH][PREF_HEIGHT][1]);
void ApplyCircleFilterGPU(BITMAPINFO *bmpInfo, void *bmpData, int radius, bool inverse);