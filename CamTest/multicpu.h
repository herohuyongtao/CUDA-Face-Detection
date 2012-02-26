//sobel.h
#pragma once
/*-------STRUCTURES---------*/
typedef struct {long rows; long cols; unsigned char* rdata; unsigned char* gdata; unsigned char* bdata; unsigned char* pdata;} sImageMultiCpu;

/*-------FUNCTIONS----------*/
long getImageInfo(BITMAP *bm, long offset, int numberOfChars);
void DetectEdgesMultiCpu(BITMAPINFO *bmpInfo, void *bmpData);