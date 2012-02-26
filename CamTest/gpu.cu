//sobel.c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "windows.h"
#include "gpu.cuh"
#include "book.h"

__global__ void sobel24BitsGpu(unsigned char *originalImageSegment, unsigned char *edgeImageSegment,
							long imageRows, long imageCols, int chunkNumber, int chunkSize)
{
	int X = threadIdx.x + chunkNumber * chunkSize;
	int Y = blockIdx.x;

	int GX[3][3];
	int GY[3][3];

	long sumX = 0;
	long sumY = 0;
	int SUM = 0;

	if (Y < imageRows && X < imageCols)
	{

		GX[0][0] = -1; GX[0][1] = 0; GX[0][2] = 1;
		GX[1][0] = -2; GX[1][1] = 0; GX[1][2] = 2;
		GX[2][0] = -1; GX[2][1] = 0; GX[2][2] = 1;

		GY[0][0] =  1; GY[0][1] =  2; GY[0][2] =  1;
		GY[1][0] =  0; GY[1][1] =  0; GY[1][2] =  0;
		GY[2][0] = -1; GY[2][1] = -2; GY[2][2] = -1;

		if(Y == 0 || Y == imageRows - 1)
		{
			SUM = 0;
		}
		else if(X == 0 || X == imageCols - 1)
		{
			SUM = 0;
		}
		else   
		{
			for(int I=-1; I<=1; I++)  
			{
				for(int J=-1; J<=1; J++)  
				{
					sumX = sumX + ( (*(originalImageSegment + X + I + 
						(Y + J)*imageCols)) * GX[I+1][J+1]);
				}
			}

			for(int I=-1; I<=1; I++)  
			{
				for(int J=-1; J<=1; J++)  
				{
					sumY = sumY + ( (*(originalImageSegment + X + I + 
						(Y + J)*imageCols)) * GY[I+1][J+1]);
				}
			}

			SUM = sqrt(pow(sumX,2.0) + pow(sumY,2.0));
		
			 if(SUM > 255) SUM = 255;
             if(SUM < 0) SUM = 0;
		
		}
		*(edgeImageSegment + X + Y*imageCols) = SUM;
	}
}

void DetectEdgesGPU(BITMAPINFO *bmpInfo, void *bmpData)
{
	unsigned char 	*dev_originalImageSegmentRed, *dev_edgeImageSegmentRed;
	unsigned char 	*dev_originalImageSegmentGreen, *dev_edgeImageSegmentGreen;
	unsigned char 	*dev_originalImageSegmentBlue, *dev_edgeImageSegmentBlue;
	unsigned long	vectorSize, paddingSize, dataLength;
	sImage originalImage, edgeImage;
	int numPadding, chunkSize, i, cont;
	long size;

	originalImage.cols = (*bmpInfo).bmiHeader.biWidth;
	originalImage.rows = (*bmpInfo).bmiHeader.biHeight;
	edgeImage.rows = originalImage.rows;
	edgeImage.cols = originalImage.cols;
	numPadding = originalImage.cols % 4;
	dataLength = (*bmpInfo).bmiHeader.biSizeImage;
	paddingSize = originalImage.rows * numPadding;
	vectorSize = (dataLength - paddingSize)/3;
	int numChunks = (originalImage.cols + 511)/ 512;
	chunkSize = 512;

	edgeImage.rdata = (unsigned char*) malloc(vectorSize*sizeof(unsigned char));
	edgeImage.gdata = (unsigned char*) malloc(vectorSize*sizeof(unsigned char));
	edgeImage.bdata = (unsigned char*) malloc(vectorSize*sizeof(unsigned char));
	edgeImage.pdata = (unsigned char*) malloc(paddingSize*sizeof(unsigned char));

	originalImage.rdata = (unsigned char*) malloc(vectorSize*sizeof(unsigned char));
	originalImage.gdata = (unsigned char*) malloc(vectorSize*sizeof(unsigned char));
	originalImage.bdata = (unsigned char*) malloc(vectorSize*sizeof(unsigned char));
	originalImage.pdata = (unsigned char*) malloc(paddingSize*sizeof(unsigned char));

	cont = 0;
	for (i=0; i <= dataLength - 3; i+=3)
	{
		*(originalImage.bdata + cont) = *(((unsigned char *)bmpData) + i);
		*(originalImage.gdata + cont) = *(((unsigned char *)bmpData) + i + 1);
		*(originalImage.rdata + cont) = *(((unsigned char *)bmpData) + i + 2);
		cont++;
	}

	dim3 dgrid (originalImage.rows, 1, 1);
	dim3 dblock (chunkSize, 1, 1);

   	HANDLE_ERROR(cudaMalloc((void**)&dev_originalImageSegmentRed, vectorSize*sizeof(unsigned char)));
  	HANDLE_ERROR(cudaMalloc((void**)&dev_edgeImageSegmentRed, vectorSize*sizeof(unsigned char)));
   	HANDLE_ERROR(cudaMalloc((void**)&dev_originalImageSegmentGreen, vectorSize*sizeof(unsigned char)));
  	HANDLE_ERROR(cudaMalloc((void**)&dev_edgeImageSegmentGreen, vectorSize*sizeof(unsigned char)));
   	HANDLE_ERROR(cudaMalloc((void**)&dev_originalImageSegmentBlue, vectorSize*sizeof(unsigned char)));
  	HANDLE_ERROR(cudaMalloc((void**)&dev_edgeImageSegmentBlue, vectorSize*sizeof(unsigned char)));

	for (int i = 0; i < numChunks; i++)
	{
		//Run algorithm for red color.
   		HANDLE_ERROR(cudaMemcpy(dev_originalImageSegmentRed, originalImage.rdata, vectorSize*sizeof(unsigned char), cudaMemcpyHostToDevice));
		sobel24BitsGpu<<<dgrid, dblock>>>(dev_originalImageSegmentRed, dev_edgeImageSegmentRed, originalImage.rows, originalImage.cols, 
			i, chunkSize);
		HANDLE_ERROR(cudaMemcpy(edgeImage.rdata, dev_edgeImageSegmentRed, vectorSize * sizeof(unsigned char), cudaMemcpyDeviceToHost));
		
		//Run algorithm for green color.
   		HANDLE_ERROR(cudaMemcpy(dev_originalImageSegmentGreen, originalImage.gdata, vectorSize*sizeof(unsigned char), cudaMemcpyHostToDevice));
		sobel24BitsGpu<<<dgrid, dblock>>>(dev_originalImageSegmentGreen, dev_edgeImageSegmentGreen, originalImage.rows, originalImage.cols,
			i, chunkSize);
		HANDLE_ERROR(cudaMemcpy(edgeImage.gdata, dev_edgeImageSegmentGreen, vectorSize * sizeof(unsigned char), cudaMemcpyDeviceToHost));

		//Run algorithm for blue color.
   		HANDLE_ERROR(cudaMemcpy(dev_originalImageSegmentBlue, originalImage.bdata, vectorSize*sizeof(unsigned char), cudaMemcpyHostToDevice));
		sobel24BitsGpu<<<dgrid, dblock>>>(dev_originalImageSegmentBlue, dev_edgeImageSegmentBlue, originalImage.rows, originalImage.cols,
			i, chunkSize);
		HANDLE_ERROR(cudaMemcpy(edgeImage.bdata, dev_edgeImageSegmentBlue, vectorSize * sizeof(unsigned char), cudaMemcpyDeviceToHost));

		cudaThreadSynchronize();
	}

	cont = 0;
	for (i=0; i <= dataLength - 3; i+=3)
	{
		int grayValue = *(edgeImage.bdata + cont) * 0.3 + *(edgeImage.gdata + cont) * 0.59 + *(edgeImage.rdata + cont) * 0.11;
		*(((unsigned char *)bmpData) + i) = grayValue;
		*(((unsigned char *)bmpData) + i + 1) = grayValue;
		*(((unsigned char *)bmpData) + i + 2) = grayValue;
		cont++;
	}

	free(edgeImage.rdata);      /* Finished with edgeImage.data */
	free(originalImage.rdata);  /* Finished with originalImage.data */
	free(edgeImage.gdata);      /* Finished with edgeImage.data */
	free(originalImage.gdata);  /* Finished with originalImage.data */
	free(edgeImage.bdata);      /* Finished with edgeImage.data */
	free(originalImage.bdata);  /* Finished with originalImage.data */
	free(edgeImage.pdata);      /* Finished with edgeImage.data */
	free(originalImage.pdata);  /* Finished with originalImage.data */
	cudaFree(dev_edgeImageSegmentRed);
	cudaFree(dev_originalImageSegmentRed);
	cudaFree(dev_edgeImageSegmentBlue);
	cudaFree(dev_originalImageSegmentBlue);
	cudaFree(dev_edgeImageSegmentGreen);
	cudaFree(dev_originalImageSegmentGreen);
}