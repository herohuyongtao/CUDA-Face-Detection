// main.c

#include <windows.h>	// the usual includes
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <process.h>
#include <malloc.h>
#include <io.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <direct.h>
#include <EXCPT.H>
#include <commctrl.h>
#include "fourier_cpu.h"

#define PREF_WIDTH 160
#define PREF_HEIGHT 120

static BITMAPINFO bmVidFormat =				// prefered format
{
	{   sizeof(BITMAPINFOHEADER),		// size
		PREF_WIDTH,			// LONG       biWidth;
		PREF_HEIGHT,			// LONG       biHeight;
		1,				// WORD       biPlanes;
		24,				// WORD       biBitCount;
		BI_RGB,				// DWORD      biCompression;
		PREF_WIDTH*PREF_HEIGHT*3,	// DWORD      biSizeImage;
		0,				// LONG       biXPelsPerMeter;
		0,				// LONG       biYPelsPerMeter;
		0,				// DWORD      biClrUsed;
		0,				// DWORD      biClrImportant;
	},
	{255, 255, 255, 0}			// color intensity 
};

HBITMAP ShowBitmap(HWND hwnd, HDC hdc)
{
	HDC memDC;
	DWORD error;
	LPCTSTR szFilename = "C:\\work\\Pesquisa CUDA\\Imagens\\input6.bmp";
	
	HBITMAP hBitmap = NULL; 
	hBitmap = (HBITMAP)LoadImage(NULL, szFilename, IMAGE_BITMAP, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE);
	
	if (hBitmap)
	{
		memDC = CreateCompatibleDC(hdc);
		SelectObject(memDC, hBitmap);
		BitBlt(hdc, 0,0,PREF_WIDTH,PREF_HEIGHT, memDC, 0, 0, SRCCOPY);

		DeleteDC(memDC);
		//DeleteObject(hBitmap);
	}
	return hBitmap;
}

void TransformBitmap(HWND hwnd, HDC hdc, HBITMAP hBitmap)
{
	HDC memDC;
	HBITMAP hTransformBitmap = CreateCompatibleBitmap(hdc, 
						bmVidFormat.bmiHeader.biWidth, 
						bmVidFormat.bmiHeader.biHeight);
	LPBYTE pAux = (LPBYTE)malloc(bmVidFormat.bmiHeader.biSizeImage*sizeof(CHAR));
	GetDIBits(hdc, hBitmap, 0, bmVidFormat.bmiHeader.biHeight, pAux, &bmVidFormat, DIB_RGB_COLORS);

	FourierTransformCPU(&bmVidFormat, pAux);

	SetDIBits(hdc, hTransformBitmap, 0, bmVidFormat.bmiHeader.biHeight, pAux, &bmVidFormat, DIB_RGB_COLORS);
	memDC = CreateCompatibleDC(hdc);
	SelectObject(memDC, hTransformBitmap);
	BitBlt(hdc, PREF_WIDTH, 0, PREF_WIDTH, PREF_HEIGHT, memDC, 0,0, SRCCOPY);
}


long FAR PASCAL cam01WndProc(HWND hwnd, UINT message, WPARAM wParam, LONG lParam)
{
    HDC hdc; 
	HBITMAP hbm; 
    PAINTSTRUCT ps;

    switch(message)			
    {
	case WM_KEYUP:
		if (wParam == 0x43)
		{
		}
		break;
   	case WM_PAINT:			
		hdc = BeginPaint(hwnd, &ps);
		hbm = ShowBitmap(hwnd, hdc);
		TransformBitmap(hwnd, hdc, hbm);
		EndPaint(hwnd, &ps);
    break;

	case WM_CLOSE:			
		DestroyWindow(hwnd);
	break;

   	case WM_DESTROY:
    	PostQuitMessage(0);	// quit the message loop
    break;

    default:
	    return DefWindowProc(hwnd, message, wParam, lParam);
    }

    return 0L;
}

int WINAPI WinMain(
    HINSTANCE  hInstance,	// handle to current instance
    HINSTANCE  hPrevInstance,	// handle to previous instance
    LPSTR  lpCmdLine,		// pointer to command line
    int  nShowCmd 		// show state of window
   )

{
	char *p;
	HWND hwnd;
   	 MSG msg;
    	WNDCLASS wndclass;

	p = "CUDA - Real Time Digital Image Processing - DFT";
	
	wndclass.style = CS_HREDRAW | CS_VREDRAW;
	wndclass.lpfnWndProc = (WNDPROC) cam01WndProc;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = NULL;					
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetSysColorBrush(COLOR_3DFACE);
	wndclass.lpszMenuName = NULL;						
	wndclass.lpszClassName = p;

	RegisterClass(&wndclass);

    
	hwnd = CreateWindow(p,		// window class name
			p,		// window caption
			WS_OVERLAPPEDWINDOW | WS_MINIMIZE,	//  window style
			0,		// x pos
			0,		// y pos
			(PREF_WIDTH * 2) + 20,		// x size
			(PREF_HEIGHT * 2) + 20,		// y size
			NULL,			// parent window handle
			NULL,			// window menu handle
			hInstance,			// prog instance handle
			NULL);			// creation parameters
	
	
		ShowWindow(hwnd, SW_SHOW); 		// window to screen
		UpdateWindow(hwnd); 			// repaint thyself

		while (GetMessage(&msg, NULL, 0, 0))	// message loop for all windows 
    		{					// belonging to this thread
			TranslateMessage(&msg);
			DispatchMessage(&msg);
    		}
		
	return 0;
}
