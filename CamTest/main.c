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

#include <vfw.h>		// this holds all the media prototypes etc

#include "video.h"		// this is our new include file 
#include "globals.h"

bool cpuOn = false;
bool multiCpuOn = false;
bool gpuOn = false;
bool edgesOn = false;
bool transformOn = false;
bool inverseOn = false;

long FAR PASCAL cam01WndProc(HWND hwnd, UINT message, WPARAM wParam, LONG lParam)
{
    HDC hdc; 
    PAINTSTRUCT ps;

    switch(message)			
    {
	case WM_KEYUP:
		if (wParam == 0x43)
		{
			if (cpuOn)
			{
				cpuOn = false;
				ClearWindowArea(hwnd, 1);
			}
			else
				cpuOn = true;
		}
		if (wParam == 0x4D)
		{
			if (multiCpuOn)
			{
				multiCpuOn = false;
				ClearWindowArea(hwnd, 2);
			}
			else
				multiCpuOn = true;
		}
		if (wParam == 0x47)
		{
			if (gpuOn)
			{
				gpuOn = false;
				ClearWindowArea(hwnd, 3);
			}
			else
				gpuOn = true;
		}
		if (wParam == 0x45)
		{
			if (edgesOn)
			{
				edgesOn = false;
				transformOn = false;
				inverseOn = false;
				ClearWindowArea(hwnd, 3);
			}
			else
				edgesOn = true;
		}
		if (wParam == 0x54)
		{
			if (transformOn)
			{
				transformOn = false;
				inverseOn = false;
				ClearWindowArea(hwnd, 3);
			}
			else
			{
				transformOn = true;
				edgesOn = true;
			}
		}
		if (wParam == 0x49)
		{
			if (inverseOn)
			{
				inverseOn = false;
				ClearWindowArea(hwnd, 3);
			}
			else
			{
				inverseOn = true;
				transformOn = true;
				edgesOn = true;
			}
		}
		break;
	case WM_TIMER:
		PollVideo();	// this call makes the cam driver get a frame
		break;

    	case WM_PAINT:			
		hdc = BeginPaint(hwnd, &ps);
		EndPaint(hwnd, &ps);
	    break;

	case WM_CLOSE:			
		DestroyWindow(hwnd);
		break;

    	case WM_DESTROY:
		KillTimer(hwnd, 1);	// bye bye timer
		StopVideoCapture();	// stop the capture
		StopVideo();		// stop the cam driver
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

	p = "CUDA - Real Time Digital Image Processing - Sobel Edge Detector";
	
	wndclass.style = CS_HREDRAW | CS_VREDRAW;
	wndclass.lpfnWndProc = (WNDPROC) cam01WndProc;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = NULL;					
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
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

		StartVideo(hwnd, 1);			// start up the cam
		StartVideoCapture(hwnd, 0, 0);	// tell where to put the capture image
		SetTimer(hwnd, 1, 100/25, NULL);	// set a timer to keep calling 


		while (GetMessage(&msg, NULL, 0, 0))	// message loop for all windows 
    		{					// belonging to this thread
			TranslateMessage(&msg);
			DispatchMessage(&msg);
    		}
		
	return 0;
}
