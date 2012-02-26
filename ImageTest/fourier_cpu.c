
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "windows.h"
#include "fourier_cpu.h"
#include <vector>

#define PREF_WIDTH 160
#define PREF_HEIGHT 120

using namespace std;

double pi = 3.141592654;
int contador;

const int N = PREF_WIDTH; //the width of the image
const int M = PREF_HEIGHT; //the height of the image

double fRe[N][M][1], fIm[N][M][1], fAmp[N][M][1]; //the signal's real part, imaginary part, and amplitude
double FRe[N][M][1], FIm[N][M][1], FAmp[N][M][1]; //the FT's real part, imaginary part and amplitude
//double FRe2[N][M][1], FIm2[N][M][1], FAmp2[N][M][1]; //the FT's real part, imaginary part and amplitude
double Gr2[N][M][1], Gi2[N][M][1];
//std::vector<double> Gr2(M * N);
//std::vector<double> Gi2(M * N); //temporary buffers

void CalcFourierTransformCPU4(unsigned char *edgeImageSegment, int xPos, int yPos, int m, int n, double *g, bool shift, bool neg128,
		  int chunkNumber, int chunkSize, int idx, int idy) //g is the image to be drawn
{
	int X = idx + chunkNumber * chunkSize;
	int Y = idy;

	if (Y < m && X < n)
	{

		int x2 = X, y2 = Y;
		//if(shift) {x2 = (X + n / 2) % n; y2 = (Y + m / 2) % m;} //Shift corners to center
		int color;
		//calculate color values out of the floating point buffer
		color = int(g[m * x2 + y2]); 

		if(neg128) color = color + 128;
		//negative colors give confusing effects so set them to 0
		if(color < 0) color = 0;
		//set color components higher than 255 to 255
		if(color > 255) color = 255;
		//plot the pixel
		*(edgeImageSegment + X * 3 + Y * n * 3) = color;//log(abs(color+1.0f))*255/(log(256.0f));
		*(edgeImageSegment + X * 3 + Y * n * 3 + 1) = color;//log(abs(color+1.0f))*255/(log(256.0f));
		*(edgeImageSegment + X * 3 + Y * n * 3 + 2) = color;//log(abs(color+1.0f))*255/(log(256.0f));
		contador++;
  }
}

void CalcFourierTransformCPU3(int n, int m, double *gAmp, double *gRe, double *gIm, long imageRows, long imageCols, int chunkNumber, int chunkSize, int idx, int idy)
{
	int X = idx + chunkNumber * chunkSize;
	int Y = idy;
  
	if (Y < imageRows && X < imageCols)
	{
		gAmp[m * X + Y] = sqrt(gRe[m * X + Y] * gRe[m * X + Y] + gIm[m * X + Y] * gIm[m * X + Y]);
	}
}

void CalcFourierTransformCPU2_1(int n, int m, bool inverse, double *gRe, double *gIm, int chunkNumber, int chunkSize, int idx, int idy)
{
	int X = idx + chunkNumber * chunkSize;
	int Y = idy;

	if (Y < m && X < n)
	{
		Gr2[X][Y][0] = Gi2[X][Y][0] = 0.0;
		for(int i = 0; i < m; i++)
		{
			double a= 2 * pi * Y * i / float(m);
			if(!inverse)a = -a;
			double ca = cos(a);
			double sa = sin(a);
			Gr2[X][Y][0] += gRe[m * X + i] * ca - gIm[m * X + i] * sa;
			Gi2[X][Y][0] += gRe[m * X + i] * sa + gIm[m * X + i] * ca;    
		}
	}
}
void CalcFourierTransformCPU2_2(int n, int m, bool inverse, double *GRe, double *GIm, int chunkNumber, int chunkSize, int idx, int idy)
{
	int X = idx + chunkNumber * chunkSize;
	int Y = idy;

	if (Y < m && X < n)
	{
		GRe[m * X + Y] = GIm[m * X + Y] = 0;
		for(int i = 0; i < n; i++)
		{
			double a = 2 * pi * X * i / float(n);
			if(!inverse)a = -a;
			double ca = cos(a);
			double sa = sin(a);
			GRe[m * X + Y] += Gr2[i][Y][0] * ca - Gi2[i][Y][0] * sa;
			GIm[m * X + Y] += Gr2[i][Y][0] * sa + Gi2[i][Y][0] * ca;
		}
		if(inverse)
		{
			GRe[m * X + Y] /= n;
			GIm[m * X + Y] /= n;
		}
		else
		{
			GRe[m * X + Y] /= m;
			GIm[m * X + Y] /= m;
		}
	}
}

void CalcFourierTransformCPU1(int n, int m, unsigned char *originalImageSegment, unsigned char *edgeImageSegment,
				int chunkNumber, int chunkSize, int idx, int idy)
{
	int X = idx + chunkNumber * chunkSize;
	int Y = idy;

	if (Y < m && X < n)
	{	
		fRe[X][Y][0] = *(originalImageSegment + X * 3 + Y * n * 3);
	}
}

void FourierTransformCPU(BITMAPINFO *bmpInfo, void *bmpData)
{
	unsigned char 	*dev_originalImageSegment, *dev_edgeImageSegment;
	unsigned long	vectorSize, paddingSize, dataLength;
	sImageSerialFourier originalImage, edgeImage;
	int numPadding, chunkSize, i, cont;
	long size;

	originalImage.cols = (*bmpInfo).bmiHeader.biWidth;
	originalImage.rows = (*bmpInfo).bmiHeader.biHeight;
	edgeImage.rows = originalImage.rows;
	edgeImage.cols = originalImage.cols;
	numPadding = originalImage.cols % 4;
	dataLength = (*bmpInfo).bmiHeader.biSizeImage;
	paddingSize = originalImage.rows * numPadding;
	vectorSize = dataLength - paddingSize;
	int numChunks = (originalImage.cols + 511)/ 512;
	chunkSize = 512;

	edgeImage.data = (unsigned char*) malloc(vectorSize*sizeof(unsigned char));
	if (edgeImage.data == NULL)
		exit;

	originalImage.data = (unsigned char*) malloc(vectorSize*sizeof(unsigned char));
	if (originalImage.data == NULL)
		exit;

	for (i=0; i < dataLength; i++)
	{
		*(originalImage.data + i) = *(((unsigned char *)bmpData) + i);
	}
	contador = 0;
   	for (int i = 0; i < numChunks; i++)
	{
		for (int j = 0; j < originalImage.rows; j++)
		{
			for (int k = 0; k < chunkSize; k++)
			{
				CalcFourierTransformCPU1(N, M, originalImage.data, edgeImage.data, i, chunkSize, k, j);
			}
		}
	}
   	for (int i = 0; i < numChunks; i++)
	{
		for (int j = 0; j < originalImage.rows; j++)
		{
			for (int k = 0; k < chunkSize; k++)
			{
				CalcFourierTransformCPU2_1(N, M, 0, fRe[0][0], fIm[0][0], i, chunkSize, k, j);
			}
		}
	}
   	for (int i = 0; i < numChunks; i++)
	{
		for (int j = 0; j < originalImage.rows; j++)
		{
			for (int k = 0; k < chunkSize; k++)
			{
				CalcFourierTransformCPU2_2(N, M, 0, FRe[0][0], FIm[0][0], i, chunkSize, k, j);
			}
		}
	}
	//inverse
	for (int i = 0; i < numChunks; i++)
	{
		for (int j = 0; j < originalImage.rows; j++)
		{
			for (int k = 0; k < chunkSize; k++)
			{
				CalcFourierTransformCPU2_1(N, M, 1, FRe[0][0], FIm[0][0], i, chunkSize, k, j);
			}
		}
	}
   	for (int i = 0; i < numChunks; i++)
	{
		for (int j = 0; j < originalImage.rows; j++)
		{
			for (int k = 0; k < chunkSize; k++)
			{
				CalcFourierTransformCPU2_2(N, M, 1, FRe[0][0], FIm[0][0], i, chunkSize, k, j);
			}
		}
	}
 
	//
   	for (int i = 0; i < numChunks; i++)
	{
		for (int j = 0; j < originalImage.rows; j++)
		{
			for (int k = 0; k < chunkSize; k++)
			{
				CalcFourierTransformCPU3(N, M, FAmp[0][0], FRe[0][0], FIm[0][0], originalImage.rows, originalImage.cols, i, chunkSize, k, j);
			}
		}
	}
   	for (int i = 0; i < numChunks; i++)
	{
		for (int j = 0; j < originalImage.rows; j++)
		{
			for (int k = 0; k < chunkSize; k++)
			{
				CalcFourierTransformCPU4(edgeImage.data, 0, 0, PREF_HEIGHT, PREF_WIDTH, FAmp[0][0], 1, 0, i, chunkSize, k, j);
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