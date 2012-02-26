//fourier_cpu.h
#pragma once
/*-------STRUCTURES---------*/
typedef struct {long rows; long cols; unsigned char* data;} sImageSerialFourier;

/*-------FUNCTIONS----------*/
long getImageInfo(BITMAP *bm, long offset, int numberOfChars);
void FourierTransformCPU(BITMAPINFO *bmpInfo, void *bmpData, bool inverse, double *img);
void ApplyCircleFilterCPU(BITMAPINFO *bmpInfo, void *bmpData, int radius, bool inverse);