//sobel.h
#pragma once
/*-------STRUCTURES---------*/
typedef struct {long rows; long cols; unsigned char* rdata; unsigned char* gdata; unsigned char* bdata; unsigned char* pdata;} sImage;

/*-------FUNCTIONS----------*/
void DetectEdgesGPU(BITMAPINFO *bmpInfo, void *bmpData);