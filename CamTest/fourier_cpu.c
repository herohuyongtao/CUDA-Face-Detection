
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "windows.h"
#include "fourier_cpu.h"
#include "globals.h"
#include <vector>

using namespace std;

double pi = 3.141592654;

const int N = PREF_WIDTH; //the width of the image
const int M = PREF_HEIGHT; //the height of the image

double fRe[N][M][1], fIm[N][M][1], fAmp[N][M][1]; //the signal's real part, imaginary part, and amplitude
double FRe[N][M][1], FAmp[N][M][1]; //the FT's real part, imaginary part and amplitude
double Gr2[N][M][1], Gi2[N][M][1];
double *FIm;

void CalcFourierTransformCPU4(bool inverse, unsigned char *edgeImageSegment, int chunkNumber, int chunkSize, int idx, int idy) //g is the image to be drawn
{
	int X = idx + chunkNumber * chunkSize;
	int Y = idy;

	if (Y < M && X < N)
	{

		int x2 = X, y2 = Y;

		//Shift corners to center
		if (!inverse)
		{
			x2 = (X + N / 2) % N; 
			y2 = (Y + M / 2) % M;
		}
		int color;
		//calculate color values out of the floating point buffer
		color = int(((double*)FRe)[M * x2 + y2]); 

		//negative colors give confusing effects so set them to 0
		if(color < 0) color = 0;
		//set color components higher than 255 to 255
		if(color > 255) color = 255;
		//plot the pixel
		*(edgeImageSegment + X * 3 + Y * N * 3) = color;//log(abs(color+1.0f))*255/(log(256.0f));
		*(edgeImageSegment + X * 3 + Y * N * 3 + 1) = color;//log(abs(color+1.0f))*255/(log(256.0f));
		*(edgeImageSegment + X * 3 + Y * N * 3 + 2) = color;//log(abs(color+1.0f))*255/(log(256.0f));
  }
}

void CalcFourierTransformCPU2_2(bool inverse, int chunkNumber, int chunkSize, int idx, int idy)
{
	int X = idx + chunkNumber * chunkSize;
	int Y = idy;

	if (X < N && Y < M)
	{
		((double*)FRe)[M * X + Y] = ((double*)FIm)[M * X + Y] = 0;
		for(int i = 0; i < N; i++)
		{
			double a = 2 * pi * X * i / float(N);
			if(!inverse)a = -a;
			double ca = cos(a);
			double sa = sin(a);
			((double*)FRe)[M * X + Y] += Gr2[i][Y][0] * ca - Gi2[i][Y][0] * sa;
			((double*)FIm)[M * X + Y] += Gr2[i][Y][0] * sa + Gi2[i][Y][0] * ca;
		}
		if(inverse)
		{
			((double*)FRe)[M * X + Y] /= N;
			((double*)FIm)[M * X + Y] /= N;
		}
		else
		{
			((double*)FRe)[M * X + Y] /= M;
			((double*)FIm)[M * X + Y] /= M;
		}
	}
}

void CalcFourierTransformCPU2_1(bool inverse, int chunkNumber, int chunkSize, int idx, int idy)
{
	int X = idx + chunkNumber * chunkSize;
	int Y = idy;

	if (X < N && Y < M)
	{
		Gr2[X][Y][0] = Gi2[X][Y][0] = 0.0;
		for(int i = 0; i < M; i++)
		{
			double a= 2 * pi * Y * i / float(M);
			if(!inverse)a = -a;
			double ca = cos(a);
			double sa = sin(a);
			if (!inverse)
			{
				Gr2[X][Y][0] += ((double*)fRe)[M * X + i] * ca - ((double*)fIm)[M * X + i] * sa;
				Gi2[X][Y][0] += ((double*)fRe)[M * X + i] * sa + ((double*)fIm)[M * X + i] * ca;    
			}
			else
			{
				Gr2[X][Y][0] += ((double*)FRe)[M * X + i] * ca - ((double*)FIm)[M * X + i] * sa;
				Gi2[X][Y][0] += ((double*)FRe)[M * X + i] * sa + ((double*)FIm)[M * X + i] * ca;    
			}
		}
	}
}

void CalcFourierTransformCPU1(bool inverse, unsigned char *originalImageSegment, unsigned char *edgeImageSegment,
				int chunkNumber, int chunkSize, int idx, int idy)
{
	int X = idx + chunkNumber * chunkSize;
	int Y = idy;
	int x2, y2;

	if (X < N && Y < M)
	{	
		if(inverse)
		{
			x2 = (X + N / 2) % N; 
			y2 = (Y + M / 2) % M;
			FRe[X][Y][0] = *(originalImageSegment + x2 * 3 + y2 * 3 * N);
		}
		else
		{
			fRe[X][Y][0] = *(originalImageSegment + X * 3 + Y * 3 * N);
			fIm[X][Y][0] = 0;
		}
	}
}

void FourierTransformCPU(BITMAPINFO *bmpInfo, void *bmpData, bool inverse, double *img)
{
	unsigned char 	*dev_originalImageSegment, *dev_edgeImageSegment;
	unsigned long	vectorSize, paddingSize, dataLength;
	sImageSerialFourier originalImage, edgeImage;
	int numPadding, chunkSize, i, j, cont;
	long size;

	originalImage.cols = (*bmpInfo).bmiHeader.biWidth;
	originalImage.rows = (*bmpInfo).bmiHeader.biHeight;
	edgeImage.rows = originalImage.rows;
	edgeImage.cols = originalImage.cols;
	numPadding = originalImage.cols % 4;
	dataLength = (*bmpInfo).bmiHeader.biSizeImage;
	paddingSize = originalImage.rows * numPadding;
	vectorSize = dataLength - paddingSize;
	chunkSize = 320;
	int numChunks = (originalImage.cols + chunkSize - 1)/ chunkSize;
	

	edgeImage.data = (unsigned char*) malloc(vectorSize*sizeof(unsigned char));
	if (edgeImage.data == NULL)
		exit;

	originalImage.data = (unsigned char*) malloc(vectorSize*sizeof(unsigned char));
	if (originalImage.data == NULL)
		exit;

	FIm = img;

	for (i=0; i < dataLength; i++)
	{
		*(originalImage.data + i) = *(((unsigned char *)bmpData) + i);
	}

   	for (int i = 0; i < numChunks; i++)
	{
		for (int j = 0; j < N; j++)
		{
			for (int k = 0; k < chunkSize; k++)
			{
				CalcFourierTransformCPU1(inverse, originalImage.data, edgeImage.data, i, chunkSize, k, j);
			}
		}
	}

   	for (int i = 0; i < numChunks; i++)
	{
		for (int j = 0; j < N; j++)
		{
			for (int k = 0; k < chunkSize; k++)
			{
				CalcFourierTransformCPU2_1(inverse, i, chunkSize, k, j);
			}
		}
	}

   	for (int i = 0; i < numChunks; i++)
	{
		for (int j = 0; j < N; j++)
		{
			for (int k = 0; k < chunkSize; k++)
			{
				CalcFourierTransformCPU2_2(inverse, i, chunkSize, k, j);
			}
		}
	}

   	for (int i = 0; i < numChunks; i++)
	{
		for (int j = 0; j < originalImage.rows; j++)
		{
			for (int k = 0; k < chunkSize; k++)
			{
				CalcFourierTransformCPU4(inverse, edgeImage.data, i, chunkSize, k, j);
			}
		}
	}

	for (i=0; i < dataLength; i++)
	{
		*(((unsigned char *)bmpData) + i) = *(edgeImage.data + i);
	}

	free(edgeImage.data);      /* Finished with edgeImage.data */
	free(originalImage.data);  /* Finished with originalImage.data */
}