
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "windows.h"
#include "fourier_multicpu.h"
#include <omp.h>
#include "globals.h"
#include <vector>

using namespace std;

double m_pi = 3.141592654;

const int N = PREF_WIDTH; //the width of the image
const int M = PREF_HEIGHT; //the height of the image

double m_fRe[N][M][1], m_fIm[N][M][1], m_fAmp[N][M][1]; //the signal's real part, imaginary part, and amplitude
double m_FRe[N][M][1], m_FAmp[N][M][1]; //the FT's real part, imaginary part and amplitude
double m_Gr2[N][M][1], m_Gi2[N][M][1]; //temporary buffers
double *m_FIm;

void CalcFourierTransformMultiCPU4(bool inverse, unsigned char *edgeImageSegment, int chunkNumber, int chunkSize, int idx, int idy) //g is the image to be drawn
{
	int X = idx + chunkNumber * chunkSize;
	int Y = idy;

	if (Y < M && X < N)
	{

		int x2 = X, y2 = Y;
		if(!inverse) 
		{
			x2 = (X + N / 2) % N; 
			y2 = (Y + M / 2) % M;
		} //Shift corners to center
		int color;
		//calculate color values out of the floating point buffer
		color = int(((double*)m_FRe)[M * x2 + y2]); 

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

void CalcFourierTransformMultiCPU2_2(bool inverse, int chunkNumber, int chunkSize, int idx, int idy)
{
	int X = idx + chunkNumber * chunkSize;
	int Y = idy;

	if (Y < M && X < N)
	{
		((double*)m_FRe)[M * X + Y] = ((double*)m_FIm)[M * X + Y] = 0;
		for(int i = 0; i < N; i++)
		{
			double a = 2 * m_pi * X * i / float(N);
			if(!inverse)a = -a;
			double ca = cos(a);
			double sa = sin(a);
			((double*)m_FRe)[M * X + Y] += m_Gr2[i][Y][0] * ca - m_Gi2[i][Y][0] * sa;
			((double*)m_FIm)[M * X + Y] += m_Gr2[i][Y][0] * sa + m_Gi2[i][Y][0] * ca;
		}
		if(inverse)
		{
			((double*)m_FRe)[M * X + Y] /= N;
			((double*)m_FIm)[M * X + Y] /= N;
		}
		else
		{
			((double*)m_FRe)[M * X + Y] /= M;
			((double*)m_FIm)[M * X + Y] /= M;
		}
	}
}

void CalcFourierTransformMultiCPU2_1(bool inverse, int chunkNumber, int chunkSize, int idx, int idy)
{
	int X = idx + chunkNumber * chunkSize;
	int Y = idy;

	if (Y < M && X < N)
	{
		m_Gr2[X][Y][0] = m_Gi2[X][Y][0] = 0;
		for(int i = 0; i < M; i++)
		{
			double a= 2 * m_pi * Y * i / float(M);
			if(!inverse)a = -a;
			double ca = cos(a);
			double sa = sin(a);
			if (!inverse)
			{
				m_Gr2[X][Y][0] += ((double*)m_fRe)[M * X + i] * ca - ((double*)m_fIm)[M * X + i] * sa;
				m_Gi2[X][Y][0] += ((double*)m_fRe)[M * X + i] * sa + ((double*)m_fIm)[M * X + i] * ca;
			}
			else
			{
				m_Gr2[X][Y][0] += ((double*)m_FRe)[M * X + i] * ca - ((double*)m_FIm)[M * X + i] * sa;
				m_Gi2[X][Y][0] += ((double*)m_FRe)[M * X + i] * sa + ((double*)m_FIm)[M * X + i] * ca;
			}
		}
	}
}

void CalcFourierTransformMultiCPU1(bool inverse, unsigned char *originalImageSegment, unsigned char *edgeImageSegment,
				int chunkNumber, int chunkSize, int idx, int idy)
{
	int X = idx + chunkNumber * chunkSize;
	int Y = idy;
	int x2, y2;

	if (Y < M && X < N)
	{	
		if(inverse)
		{
			x2 = (X + N / 2) % N; 
			y2 = (Y + M / 2) % M;
			m_FRe[X][Y][0] = *(originalImageSegment + x2 * 3 + y2 * 3 * N);
		}
		else
		{
			m_fRe[X][Y][0] = *(originalImageSegment + X * 3 + Y * 3 * N);
			m_fIm[X][Y][0] = 0;
		}
	}
}

void FourierTransformMultiCPU(BITMAPINFO *bmpInfo, void *bmpData, bool inverse, double *img)
{
	unsigned char 	*dev_originalImageSegment, *dev_edgeImageSegment;
	unsigned long	vectorSize, paddingSize, dataLength;
	sImageMultiCpuFourier originalImage, edgeImage;
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

	m_FIm = img;

	for (i=0; i < dataLength; i++)
	{
		*(originalImage.data + i) = *(((unsigned char *)bmpData) + i);
	}

	omp_set_num_threads(8);

   	for (int i = 0; i < numChunks; i++)
	{
		for (int j = 0; j < originalImage.rows; j++)
		{
			#pragma omp parallel for 
			for (int k = 0; k < chunkSize; k++)
			{
				CalcFourierTransformMultiCPU1(inverse, originalImage.data, edgeImage.data, i, chunkSize, k, j);
			}
		}
	}
   	for (int i = 0; i < numChunks; i++)
	{
		for (int j = 0; j < originalImage.rows; j++)
		{
			#pragma omp parallel for 
			for (int k = 0; k < chunkSize; k++)
			{
				CalcFourierTransformMultiCPU2_1(inverse, i, chunkSize, k, j);
			}
		}
	}
   	for (int i = 0; i < numChunks; i++)
	{
		for (int j = 0; j < originalImage.rows; j++)
		{
			#pragma omp parallel for 
			for (int k = 0; k < chunkSize; k++)
			{
				CalcFourierTransformMultiCPU2_2(inverse, i, chunkSize, k, j);
			}
		}
	}
   	for (int i = 0; i < numChunks; i++)
	{
		for (int j = 0; j < originalImage.rows; j++)
		{
			#pragma omp parallel for 
			for (int k = 0; k < chunkSize; k++)
			{
				CalcFourierTransformMultiCPU4(inverse, edgeImage.data, i, chunkSize, k, j);
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

void ApplyCircleFilterMultiCpu(BITMAPINFO *bmpInfo, void *bmpData, int radius, bool inverse)
{
	int width = (*bmpInfo).bmiHeader.biWidth;
	int height = (*bmpInfo).bmiHeader.biHeight;

	int centerX = width/2;
	int centerY = height/2;
	int i = 0, j = 0;

	for (i = 0; i < width; i++)
	{
		#pragma omp parallel for 
		for (j = 0; j < height; j++)
		{
			int distance = sqrt(pow(i - centerX, 2.0) + pow(j - centerY, 2.0));
			if (inverse)
			{
				if (distance <= radius)
				{
					*(((unsigned char *)bmpData) + i * 3 + j * width * 3) = 0;
					*(((unsigned char *)bmpData) + i * 3 + j * width * 3 + 1) = 0;
					*(((unsigned char *)bmpData) + i * 3 + j * width * 3 + 2) = 0;
				}
			}
			else
			{
				if (distance > radius)
				{
					*(((unsigned char *)bmpData) + i * 3 + j * width * 3) = 0;
					*(((unsigned char *)bmpData) + i * 3 + j * width * 3 + 1) = 0;
					*(((unsigned char *)bmpData) + i * 3 + j * width * 3 + 2) = 0;
				}
			}
		}
	}
}