//fourier_multicpu.h
#pragma once
/*-------STRUCTURES---------*/
typedef struct {long rows; long cols; unsigned char* data;} sImageMultiCpuFourier;

/*-------FUNCTIONS----------*/
long getImageInfo(BITMAP *bm, long offset, int numberOfChars);
void FourierTransformMultiCPU(BITMAPINFO *bmpInfo, void *bmpData, bool inverse, double *img);