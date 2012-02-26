//fourier_gpu.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "windows.h"
//#include "book.h"
#include "globals.h"
#include "fourier_gpu.cuh"

using namespace std;

__device__ __constant__ float gpu_pi = 3.1415926535897932384626433832795;

const int N = PREF_WIDTH; //the width of the image
const int M = PREF_HEIGHT; //the height of the image

__device__ float gpu_Gr2[N][M][1];
__device__ float gpu_Gi2[N][M][1]; //temporary buffers

__device__ float gpu_fRe[N][M][1], gpu_fIm[N][M][1]; //the signal's real part, imaginary part, and amplitude
__device__ float gpu_FRe[N][M][1], gpu_FAmp[N][M][1]; //the FT's real part, imaginary part and amplitude
__device__ float gpu_FIm[N][M][1];

__global__ void CalcFourierTransformGPU4(bool inverse, unsigned char *edgeImageSegment, int chunkNumber, int chunkSize) //g is the image to be drawn
{
	int X = threadIdx.x + chunkNumber * chunkSize;
	int Y = blockIdx.x;

	if (X < N && Y < M)
	{

		int x2 = X, y2 = Y;
		if(!inverse) 
		{
			x2 = (X + N / 2) % N; 
			y2 = (Y + M / 2) % M;
		} //Shift corners to center
		int color;
		////calculate color values out of the floating point buffer
		color = int(((float*)gpu_FRe)[M * x2 + y2]); 
		
		////negative colors give confusing effects so set them to 0
		if(color < 0) color = 0;
		////set color components higher than 255 to 255
		if(color > 255) color = 255;
		//plot the pixel
		*(edgeImageSegment + X * 3 + Y * N * 3) = color;//log(abs(color+1.0f))*255/(log(256.0f));
		*(edgeImageSegment + X * 3 + Y * N * 3 + 1) = color;//log(abs(color+1.0f))*255/(log(256.0f));
		*(edgeImageSegment + X * 3 + Y * N * 3 + 2) = color;//log(abs(color+1.0f))*255/(log(256.0f));
  }
}

__global__ void CalcFourierTransformGPU2_2(bool inverse, int chunkNumber, int chunkSize)
{
	int X = threadIdx.x + chunkNumber * chunkSize;
	int Y = blockIdx.x;

	if (X < N && Y < M)
	{
		((float*)gpu_FRe)[M  * X + Y] = 0;
		((float*)gpu_FIm)[M  * X + Y] = 0;
		for(int i = 0; i < N; i++)
		{
			float a = 2 * gpu_pi * X * i / float(N);
			if(!inverse)a = -a;
			float ca = cos(a);
			float sa = sin(a);
			((float*)gpu_FRe)[M  * X + Y] += gpu_Gr2[i][Y][0] * ca - gpu_Gi2[i][Y][0] * sa;
			((float*)gpu_FIm)[M  * X + Y] += gpu_Gr2[i][Y][0] * sa + gpu_Gi2[i][Y][0] * ca;
		}
		if(inverse)
		{
			((float*)gpu_FRe)[M  * X + Y] /= N;
			((float*)gpu_FIm)[M  * X + Y] /= N;
		}
		else
		{
			((float*)gpu_FRe)[M  * X + Y] /= M;
			((float*)gpu_FIm)[M  * X + Y] /= M;
		}
	}
}

__global__ void CalcFourierTransformGPU2_1(bool inverse, int chunkNumber, int chunkSize)
{
	int X = threadIdx.x + chunkNumber * chunkSize;
	int Y = blockIdx.x;
	
	if (X < N && Y < M)
	{
		gpu_Gr2[X][Y][0] = 0;
		gpu_Gi2[X][Y][0] = 0;
		for(int i = 0; i < M; i++)
		{
			float a= 2 * gpu_pi * Y * i / float(M);
			if(!inverse)a = -a;
			float ca = cos(a);
			float sa = sin(a);
			if (!inverse)
			{
			  gpu_Gr2[X][Y][0] += ((float*)gpu_fRe)[M * X + i] * ca - ((float*)gpu_fIm)[M * X + i] * sa;
			  gpu_Gi2[X][Y][0] += ((float*)gpu_fRe)[M * X + i] * sa + ((float*)gpu_fIm)[M * X + i] * ca;
			}
			else
			{
			  gpu_Gr2[X][Y][0] += ((float*)gpu_FRe)[M * X + i] * ca - ((float*)gpu_FIm)[M * X + i] * sa;
			  gpu_Gi2[X][Y][0] += ((float*)gpu_FRe)[M * X + i] * sa + ((float*)gpu_FIm)[M * X + i] * ca;
			}
		}
	}
}

__global__ void CalcFourierTransformGPU1(bool inverse, unsigned char *originalImageSegment, int chunkNumber, int chunkSize)
{
	int X = threadIdx.x + chunkNumber * chunkSize;
	int Y = blockIdx.x;
	int x2, y2;

	if (X < N && Y < M)
	{	
		if (inverse)
		{
			x2 = (X + N / 2) % N; 
			y2 = (Y + M / 2) % M;
			gpu_FRe[X][Y][0] = *(originalImageSegment + x2 * 3 + y2 * 3 * N);
		}
		else
		{
			gpu_fRe[X][Y][0] = *(originalImageSegment + X * 3 + Y * 3 * N);
			gpu_fIm[X][Y][0] = 0;
		}
	}
}

void checkError()
{
	 // check for error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
	  char message[255];
	  sprintf(message, "CUDA error: %s\n", cudaGetErrorString(error));
	  MessageBox(NULL, message, "Error", MB_OK);
	  exit(-1);
  }
}

void FourierTransformGPU(BITMAPINFO *bmpInfo, void *bmpData, bool inverse, float img[PREF_WIDTH][PREF_HEIGHT][1])
{
	char message[255];
	unsigned char 	*dev_originalImageSegment, *dev_edgeImageSegment;
	unsigned long	vectorSize, paddingSize, dataLength;
	sImageFourier originalImage, edgeImage;
	int numPadding, chunkSize, i, cont;
	long size;

	originalImage.cols = (*bmpInfo).bmiHeader.biWidth;
	originalImage.rows = (*bmpInfo).bmiHeader.biHeight;
	edgeImage.rows = originalImage.rows;
	edgeImage.cols = originalImage.cols;
	numPadding = originalImage.cols % 4;
	dataLength = (*bmpInfo).bmiHeader.biSizeImage;
	paddingSize = originalImage.rows * numPadding;
	vectorSize = (dataLength - paddingSize);
	chunkSize = 320;
	int numChunks = (originalImage.cols + chunkSize - 1)/ chunkSize;

	edgeImage.data = (unsigned char*) malloc(vectorSize*sizeof(unsigned char));
	originalImage.data = (unsigned char*) malloc(vectorSize*sizeof(unsigned char));

	for (i=0; i < dataLength; i++)
	{
		*(originalImage.data + i) = *(((unsigned char *)bmpData) + i);
	}

	dim3 dgrid (originalImage.rows, 1, 1);
	dim3 dblock (chunkSize, 1, 1);

   	cudaMalloc((void**)&dev_originalImageSegment, vectorSize*sizeof(unsigned char));
  	cudaMalloc((void**)&dev_edgeImageSegment, vectorSize*sizeof(unsigned char));
	
	cudaMemcpy(dev_originalImageSegment, originalImage.data, vectorSize*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_FIm, img, N*M*sizeof(float), cudaMemcpyHostToDevice);
	
	for (int i = 0; i < numChunks; i++)
	{
		CalcFourierTransformGPU1<<<dgrid, dblock>>>(inverse, dev_originalImageSegment, i, chunkSize);
		cudaThreadSynchronize();
		
		CalcFourierTransformGPU2_1<<<dgrid, dblock>>>(inverse, i, chunkSize);
		cudaThreadSynchronize();
		
		CalcFourierTransformGPU2_2<<<dgrid, dblock>>>(inverse, i, chunkSize);
		cudaThreadSynchronize();

		CalcFourierTransformGPU4<<<dgrid, dblock>>>(inverse, dev_edgeImageSegment, i, chunkSize);
		cudaThreadSynchronize();
	}
	cudaMemcpy(edgeImage.data, dev_edgeImageSegment, vectorSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMemcpy(img, gpu_FIm, N*M*sizeof(float), cudaMemcpyDeviceToHost);	
	for (i=0; i < dataLength; i++)
	{
		*(((unsigned char *)bmpData) + i) = *(edgeImage.data + i);
	}

	free(edgeImage.data);      /* Finished with edgeImage.data */
	free(originalImage.data);  /* Finished with originalImage.data */
	cudaFree(dev_edgeImageSegment);
	cudaFree(dev_originalImageSegment);
}

__global__ void CircleFilterKernel(void *bmpData, int radius, bool inverse, int chunkNumber, int chunkSize)
{
	int X = threadIdx.x + chunkNumber * chunkSize;
	int Y = blockIdx.x;

	if (X < N && Y < M)
	{	
		int centerX = N/2;
		int centerY = M/2;

		int distance = sqrt(pow(X - centerX, 2.0) + pow(Y - centerY, 2.0));
		if (inverse)
		{
			if (distance <= radius)
			{
				*(((unsigned char *)bmpData) + X * 3 + Y * N * 3) = 0;
				*(((unsigned char *)bmpData) + X * 3 + Y * N * 3 + 1) = 0;
				*(((unsigned char *)bmpData) + X * 3 + Y * N * 3 + 2) = 0;
			}
		}
		else
		{
			if (distance > radius)
			{
				*(((unsigned char *)bmpData) + X * 3 + Y * N * 3) = 0;
				*(((unsigned char *)bmpData) + X * 3 + Y * N * 3 + 1) = 0;
				*(((unsigned char *)bmpData) + X * 3 + Y * N * 3 + 2) = 0;
			}
		}
	}
}

void ApplyCircleFilterGPU(BITMAPINFO *bmpInfo, void *bmpData, int radius, bool inverse)
{
	char message[255];
	unsigned char 	*dev_originalImageSegment;
	unsigned long	vectorSize, paddingSize, dataLength;
	int numPadding, chunkSize, i, cont;
	long size;

	numPadding = N % 4;
	dataLength = (*bmpInfo).bmiHeader.biSizeImage;
	paddingSize = M * numPadding;
	vectorSize = (dataLength - paddingSize);
	chunkSize = 320;
	int numChunks = (N + chunkSize - 1)/ chunkSize;

	dim3 dgrid (M, 1, 1);
	dim3 dblock (chunkSize, 1, 1);

   	cudaMalloc((void**)&dev_originalImageSegment, vectorSize*sizeof(unsigned char));
	
	cudaMemcpy(dev_originalImageSegment, bmpData, vectorSize*sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	for (int i = 0; i < numChunks; i++)
	{
		CircleFilterKernel<<<dgrid, dblock>>>(dev_originalImageSegment, radius, inverse, i, chunkSize);
		cudaThreadSynchronize();
	}
	cudaMemcpy(bmpData, dev_originalImageSegment, vectorSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	
	cudaFree(dev_originalImageSegment);
}

