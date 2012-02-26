//sobel.c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "windows.h"
#include "cpu.h"

void sobel24BitsCpu(unsigned char *originalImageSegment, unsigned char *edgeImageSegment,
				long imageRows, long imageCols, int chunkNumber, int chunkSize, int idx, int idy)
{
   /*---------------------------------------------------
		SOBEL ALGORITHM STARTS HERE
   ---------------------------------------------------*/
	int X = idx + chunkNumber * chunkSize;
	int Y = idy;

	int GX[3][3];
	int GY[3][3];

	long sumX = 0;
	long sumY = 0;
	int SUM = 0;

	if (Y < imageRows && X < imageCols)
	{
	 /* image boundaries */

		/* 3x3 GX Sobel mask.  Ref: www.cee.hw.ac.uk/hipr/html/sobel.html */
		GX[0][0] = -1; GX[0][1] = 0; GX[0][2] = 1;
		GX[1][0] = -2; GX[1][1] = 0; GX[1][2] = 2;
		GX[2][0] = -1; GX[2][1] = 0; GX[2][2] = 1;

		/* 3x3 GY Sobel mask.  Ref: www.cee.hw.ac.uk/hipr/html/sobel.html */
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
		/* Convolution starts here */
		else   
		{
			/*-------X GRADIENT APPROXIMATION------*/
			for(int I=-1; I<=1; I++)  
			{
				for(int J=-1; J<=1; J++)  
				{
					sumX = sumX + ( (*(originalImageSegment + X + I + 
						(Y + J)*imageCols)) * GX[I+1][J+1]);
				}
			}

			/*-------Y GRADIENT APPROXIMATION-------*/
			for(int I=-1; I<=1; I++)  
			{
				for(int J=-1; J<=1; J++)  
				{
					sumY = sumY + ( (*(originalImageSegment + X + I + 
						(Y + J)*imageCols)) * GY[I+1][J+1]);
				}
			}

			///*---GRADIENT MAGNITUDE APPROXIMATION (Myler p.218)----*/
			SUM = sqrt(pow(sumX,2.0) + pow(sumY,2.0));
		
			 if(SUM > 255) 
				 SUM = 255;
             if(SUM < 0) 
				 SUM = 0;
		}
		*(edgeImageSegment + X + Y*imageCols) = SUM;
	}
}



void DetectEdgesCpu(BITMAPINFO *bmpInfo, void *bmpData)
{
	unsigned char 	*dev_originalImageSegment, *dev_edgeImageSegment;
	unsigned long	vectorSize, paddingSize, dataLength;
	sImageSerial originalImage, edgeImage;
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
	if (edgeImage.rdata == NULL)
		exit;
	edgeImage.gdata = (unsigned char*) malloc(vectorSize*sizeof(unsigned char));
	if (edgeImage.gdata == NULL)
		exit;
	edgeImage.bdata = (unsigned char*) malloc(vectorSize*sizeof(unsigned char));
	if (edgeImage.bdata == NULL)
		exit;
	edgeImage.pdata = (unsigned char*) malloc(paddingSize*sizeof(unsigned char));

	originalImage.rdata = (unsigned char*) malloc(vectorSize*sizeof(unsigned char));
	if (originalImage.rdata == NULL)
		exit;
	originalImage.gdata = (unsigned char*) malloc(vectorSize*sizeof(unsigned char));
	if (originalImage.gdata == NULL)
		exit;
	originalImage.bdata = (unsigned char*) malloc(vectorSize*sizeof(unsigned char));
	if (originalImage.bdata == NULL)
		exit;
	originalImage.pdata = (unsigned char*) malloc(paddingSize*sizeof(unsigned char));

	cont = 0;
	for (i=0; i <= dataLength - 3; i+=3)
	{
		*(originalImage.bdata + cont) = *(((unsigned char *)bmpData) + i);
		*(originalImage.gdata + cont) = *(((unsigned char *)bmpData) + i + 1);
		*(originalImage.rdata + cont) = *(((unsigned char *)bmpData) + i + 2);
		cont++;
	}

   	for (int i = 0; i < numChunks; i++)
	{
		for (int j = 0; j < originalImage.rows; j++)
		{
			for (int k = 0; k < chunkSize; k++)
			{
				//Run algorithm for red color.
				sobel24BitsCpu(originalImage.rdata, edgeImage.rdata, originalImage.rows, originalImage.cols, i, chunkSize, k, j);
				//Run algorithm for green color.
				sobel24BitsCpu(originalImage.gdata, edgeImage.gdata, originalImage.rows, originalImage.cols, i, chunkSize, k, j);
				//Run algorithm for blue color.
				sobel24BitsCpu(originalImage.bdata, edgeImage.bdata, originalImage.rows, originalImage.cols, i, chunkSize, k, j);
			}
		}
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
}