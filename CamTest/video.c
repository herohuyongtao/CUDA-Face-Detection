// video.c

#include <windows.h>
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

#include <vfw.h>

#include "video.h"
#include "globals.h"
#include "gpu.cuh"
#include "cpu.h"
#include "multicpu.h"
#include "fourier_cpu.h"
#include "fourier_multicpu.h"
#include "fourier_gpu.cuh"

// a debugging caste I use
#define AddStrToVizWindow(x) MessageBox(NULL, x, "Video", MB_OK)


// VIDEO_FRAME target values
#define VF_USER		0		// update user_names video image
					

// VIDEO_FRAME compression values
#define VF_NO_COMPRESSION	0

static HWND hwVidParent = NULL;
static HWND hWndC = NULL;
static int gdwFrameNum = 0;
static int capOn = 0;

static CAPDRIVERCAPS CapDrvCaps; 
static CAPSTATUS CapStatus;
static int useDialog = 0;

static int csInited = 0;
static CRITICAL_SECTION csVideo;

#define DISPLAY_WIDTH PREF_WIDTH
#define DISPLAY_HEIGHT ((DISPLAY_WIDTH / 4) * 3)

static int v_prev = 0;
static int v_width = PREF_WIDTH;
static int v_height = PREF_HEIGHT;

static BITMAPINFO bmVidFormat =				// prefered format
{
	{    sizeof(BITMAPINFOHEADER),		// size
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

static BITMAPINFO *pVidFormat = NULL;	// will point to final format accepted 
static int format_len = 0;		// by driver
static int frameCPU = 0;
static int frameMULTICPU = 0;
static int frameGPU = 0;
static clock_t tempoCPU, tempobaseCPU;
static clock_t tempoMULTICPU, tempobaseMULTICPU;
static clock_t tempoGPU, tempobaseGPU;
static HWND hwDisplayWin = NULL;
static short dwXpos;
static short dwYpos;



// **********************************************************************

#define VID_STACK_SIZE 2 // stack of image data waiting to be read - keep small
static unsigned char *vid_data_stack[VID_STACK_SIZE];
static int vid_data_siz[VID_STACK_SIZE];
static int vid_head = 0;
static int vid_tail = 0;


// **********************************************************************************
static void DrawFPS(HDC hdc, double fps, double fpsCompare) 
{
	char* saida = (char*)malloc(255);
	char* percent = (char*)malloc(255);
	sprintf(saida, "TIME:%2.3f\0", fps);
	TextOut(hdc, 10, 10, (LPCSTR)saida, 10);

	if (fps > 0 && fpsCompare > 0)
	{
		sprintf(percent, "X:%2.3f\0", fpsCompare/fps);
		TextOut(hdc, 10, 30, (LPCSTR)percent, 8);
	}
}

static void DrawBitmap(HDC hdc,		// draw bit map func to display image
			short xstart,
			short ystart,
			HBITMAP hBitmap,
			int xoffset,
			int yoffset,
			double fps,
			double fpsCompare)
{
	BITMAP bm;
	HDC	hMemDC;
	POINT pt;
	if (hBitmap)
	{
		hMemDC = CreateCompatibleDC(hdc);
		SelectObject(hMemDC, hBitmap);
		GetObject(hBitmap, sizeof(BITMAP), (LPSTR) &bm);
		DrawFPS(hMemDC, fps, fpsCompare);
		pt.x = bm.bmWidth;
		pt.y = bm.bmHeight;
		BitBlt(hdc, xstart + xoffset, ystart + yoffset, pt.x, pt.y, hMemDC, 0,0, SRCCOPY);
		//StretchBlt(hdc, xstart + DISPLAY_WIDTH, ystart, DISPLAY_WIDTH, DISPLAY_HEIGHT, 
		//	hMemDC, xstart, ystart, pt.x, pt.y, SRCCOPY);
		DeleteDC(hMemDC);
	}
}

void ClearWindowArea(HWND hWnd, int windowNumber)
{
	if (hWnd)
	{
		int xOffSet = (DISPLAY_WIDTH * (windowNumber % 2));
		int yOffSet = (DISPLAY_HEIGHT * ((windowNumber)/2));
		POINT top;
		POINT bottom;
		RECT rect;
		top.x = xOffSet;
		top.y = yOffSet;
		bottom.x = DISPLAY_WIDTH + xOffSet;
		bottom.y = DISPLAY_HEIGHT + yOffSet;
		ClientToScreen(hWnd, &top);
		ClientToScreen(hWnd, &bottom);
		SetRect(&rect, top.x, top.y, bottom.x, bottom.y);
		HDC hdc = GetWindowDC(hWnd);
		HBRUSH hBrush = NULL;
		if (hdc)
		{
			hBrush = CreateSolidBrush(RGB(255,255,255));
		}
		if (hBrush)
		{
			SelectObject(hdc, hBrush);
			FillRect(hdc, &rect, hBrush);
			DeleteObject(hBrush);
		}
	}
}

LRESULT CALLBACK capVideoStreamCallback(HWND hWnd, LPVIDEOHDR lpVHdr)
{

    if (!hwVidParent) 
        return FALSE; 


	if (capOn)
	{
		if (hwDisplayWin)
		{
			HBITMAP hbCpu, hbMultiCpu, hbGpu;
			HDC hdc;
			hdc = GetDC(hwDisplayWin);
			int pSize = pVidFormat->bmiHeader.biSizeImage * sizeof(CHAR);
			static double fpsCPU = 0;
			static double fpsMULTICPU = 0;
			static double fpsGPU = 0;
			double FIm[PREF_WIDTH][PREF_HEIGHT][1];
			float fFIm[PREF_WIDTH][PREF_HEIGHT][1];
			if (cpuOn)
			{
				hbCpu = CreateCompatibleBitmap(hdc, 
						pVidFormat->bmiHeader.biWidth, 
						pVidFormat->bmiHeader.biHeight);
				LPBYTE pAuxCpu = NULL;
				pAuxCpu = (LPBYTE)malloc(sizeof(CHAR)*pSize);
				memcpy(pAuxCpu, lpVHdr->lpData, pSize);
		 		tempobaseCPU = clock();
				if (edgesOn)
					DetectEdgesCpu(pVidFormat, pAuxCpu);
				if (transformOn)
					FourierTransformCPU(pVidFormat, pAuxCpu, false, &FIm[0][0][0]);
				if (inverseOn)
					FourierTransformCPU(pVidFormat, pAuxCpu, true, &FIm[0][0][0]);
				tempoCPU = clock();
				fpsCPU = (tempoCPU - tempobaseCPU)/((DOUBLE)CLOCKS_PER_SEC);
				SetDIBits( hdc, hbCpu, 0, pVidFormat->bmiHeader.biHeight, pAuxCpu,	pVidFormat,	DIB_RGB_COLORS);
				DrawBitmap(hdc, dwXpos, dwYpos, hbCpu, DISPLAY_WIDTH, 0, fpsCPU, fpsCPU); 
				DeleteObject(hbCpu);
			}
			else
			{
				fpsCPU = 0;
			}

			if (multiCpuOn)
			{
				hbMultiCpu = CreateCompatibleBitmap(hdc, 
						pVidFormat->bmiHeader.biWidth, 
						pVidFormat->bmiHeader.biHeight);
				LPBYTE pAuxMultiCpu = NULL;
				pAuxMultiCpu = (LPBYTE)malloc(sizeof(CHAR)*pSize);
				memcpy(pAuxMultiCpu, lpVHdr->lpData, pSize);
		 		tempobaseMULTICPU = clock();
				if (edgesOn)
					DetectEdgesMultiCpu(pVidFormat, pAuxMultiCpu);
				if (transformOn)
					FourierTransformMultiCPU(pVidFormat, pAuxMultiCpu, false, &FIm[0][0][0]);
				if (inverseOn)
					FourierTransformMultiCPU(pVidFormat, pAuxMultiCpu, true, &FIm[0][0][0]);
				tempoMULTICPU = clock();
				fpsMULTICPU = (tempoMULTICPU - tempobaseMULTICPU)/((double)CLOCKS_PER_SEC);
				SetDIBits( hdc, hbMultiCpu, 0, pVidFormat->bmiHeader.biHeight, pAuxMultiCpu,	pVidFormat,	DIB_RGB_COLORS);
				DrawBitmap(hdc, dwXpos, dwYpos, hbMultiCpu, 0, DISPLAY_HEIGHT, fpsMULTICPU, fpsCPU); 
				DeleteObject(hbMultiCpu);
			}

			if (gpuOn)
			{
				hbGpu = CreateCompatibleBitmap(hdc, 
						pVidFormat->bmiHeader.biWidth, 
						pVidFormat->bmiHeader.biHeight);
				LPBYTE pAuxGpu = NULL;
				pAuxGpu = (LPBYTE)malloc(sizeof(CHAR)*pSize);
				memcpy(pAuxGpu, lpVHdr->lpData, pSize);
				tempobaseGPU = clock();
				if (edgesOn)
					DetectEdgesGPU(pVidFormat, pAuxGpu);
				if (transformOn)
					FourierTransformGPU(pVidFormat, pAuxGpu, false, fFIm);
				if (inverseOn)
					FourierTransformGPU(pVidFormat, pAuxGpu, true, fFIm);
				tempoGPU = clock();
				fpsGPU = (tempoGPU - tempobaseGPU)/((double)CLOCKS_PER_SEC);
				SetDIBits( hdc, hbGpu, 0, pVidFormat->bmiHeader.biHeight, pAuxGpu,	pVidFormat,	DIB_RGB_COLORS);
				DrawBitmap(hdc, dwXpos, dwYpos, hbGpu, DISPLAY_WIDTH, DISPLAY_HEIGHT, fpsGPU, fpsCPU); 
				DeleteObject(hbGpu);
			}

			ReleaseDC(hwDisplayWin, hdc);
		}
	}

    return (LRESULT) TRUE ; 
}
LRESULT CALLBACK capControlCallback( HWND hWnd,  int nState )
{
	return TRUE;
}

LRESULT CALLBACK capVideoYield(HWND hWnd)
{
  	MSG msg; 
	BOOL ret; 

	/* get the next message, if any */ 
	ret = (BOOL) PeekMessage(&msg, NULL, 0, 0, PM_REMOVE); 
 
	/* if we got one, process it */ 
	if (ret)
	{ 
		TranslateMessage(&msg); 
		DispatchMessage(&msg); 
	} 
 
	/* TRUE if we got a message */ 
	return ret; 
 
}

// ********************************************************************
// ********************************************************************
// ********************************************************************
// ********************************************************************

void StartVideoCapture(HWND hwShow, int xpos, int ypos)
{
	capOn = 1;
	hwDisplayWin = hwShow;
	dwXpos = xpos;
	dwYpos = ypos;

}
void StopVideoCapture()
{
	capOn = 0;
}


void StartVideo(HWND hwParent, int Visible)
{
	int l, x;
	char name[128], ver[128];
	CAPTUREPARMS cp;
	BITMAPINFO *pdebug; // for viewing under debug only
	DWORD dwStyle;

	memset(name,0, sizeof(name));
	memset(ver,0, sizeof(ver));
	hwVidParent = hWndC = NULL;
	pVidFormat = NULL;
	format_len = 0;

	if (csInited == 0)
	{
		InitializeCriticalSection(&csVideo);
		csInited = 1;
	}

	if (!capGetDriverDescription(0, name, sizeof(name), ver, sizeof(ver))) 
	{
		MessageBox(hwParent, "No video-cam driver", "Video", MB_OK);
		return;
	}
	
	if (strlen(name) == 0)
	{
		MessageBox(hwParent, "No video-cam driver named", "Video", MB_OK);
		return;
	}

	/*char* driverName = new char[256];
	sprintf(driverName,"Using video-cam driver %s", name);
	AddStrToVizWindow(driverName);*/

	hwVidParent = hwParent;
	gdwFrameNum = 1;

	dwStyle = WS_CHILD | WS_EX_NOPARENTNOTIFY | WS_EX_TOPMOST | WS_BORDER ;
	if (Visible)
		dwStyle |= WS_VISIBLE;

	hWndC = capCreateCaptureWindow (	
					"Broadcasting", 
					dwStyle,	
					dwXpos, 
					dwYpos, 
					DISPLAY_WIDTH ,
					DISPLAY_HEIGHT,
					hwVidParent, 
					1);

	if (hWndC)
	{
		int ok = capDriverConnect(hWndC, 0);

		ok = capDriverGetCaps(hWndC, &CapDrvCaps, sizeof (CAPDRIVERCAPS));


		if (useDialog)	// left in and commented out for you to play with
		{
							// Video source dialog box. 
			if (CapDrvCaps.fHasDlgVideoSource)	
				capDlgVideoSource(hWndC); 
 							// Video format dialog box. 
			if (CapDrvCaps.fHasDlgVideoFormat) 
			{
				capDlgVideoFormat(hWndC); 
							// Are there new image dimensions?
				capGetStatus(hWndC, &CapStatus, sizeof (CAPSTATUS));
							// If so, notify the parent of 
							// a size change.
			}
							// Video display dialog box. 
			if (CapDrvCaps.fHasDlgVideoDisplay)
				capDlgVideoDisplay(hWndC); 
		}

		if (capSetVideoFormat(hWndC, &bmVidFormat, sizeof(bmVidFormat)) 
			== FALSE)
		{
			AddStrToVizWindow("Video format rejected");
			l = capGetVideoFormatSize(hWndC); 
			if (l > 0)
			{
				pVidFormat = (BITMAPINFO *)calloc(1, l);
				if (pVidFormat)
				{
					format_len = l;
					x =  capGetVideoFormat(hWndC, pVidFormat, l);
					if (x <= 0)
					{
						free(pVidFormat);
						pVidFormat = NULL;
						format_len = 0;
						AddStrToVizWindow("Video format NOT retrieved");
					} else
						AddStrToVizWindow("Video format retrieved");
				} else
					AddStrToVizWindow("Video format calloc() failed");
			} else
				AddStrToVizWindow("Video format size return 0");

		} else
		{
			pVidFormat = &bmVidFormat;
			format_len = sizeof(bmVidFormat);
			//AddStrToVizWindow("Prefered video format accepted");
		}

		pdebug = &bmVidFormat;
		pdebug = pVidFormat;

		capSetCallbackOnVideoStream(hWndC, capVideoStreamCallback);
		capSetCallbackOnFrame(hWndC, capVideoStreamCallback);

		capSetCallbackOnYield(hWndC, capVideoYield); 
		capSetCallbackOnCapControl(hWndC, capControlCallback);

		capPreviewRate(hWndC, 40);
		capPreview(hWndC, FALSE);
		capOverlay(hWndC, FALSE);		// enabling overlay disables preview
		capPreviewScale(hWndC, TRUE);	// images scaled

		capCaptureGetSetup(hWndC, &cp, sizeof(cp));
		cp.dwRequestMicroSecPerFrame = 33333;	// 30 frames per sec

		cp.wPercentDropForError = 100;
		cp.fYield = FALSE;
		cp.fCaptureAudio = FALSE;
		cp.vKeyAbort = 0;
		cp.fAbortLeftMouse = FALSE;
		cp.fAbortRightMouse = FALSE;
		cp.fLimitEnabled = FALSE;

		capCaptureSetSetup(hWndC, &cp, sizeof(cp)); 
	}

}

void PollVideo()
{
	capGrabFrame(hWndC);	// gets the frame in the stack which 
				// causes the callback function to do its thing
}

void StopVideo()
{
	DWORD t = GetTickCount() + 1000; // wait a sec

	if (hWndC)
	{
		capCaptureStop(hWndC); 
		PostMessage(hWndC, WM_CLOSE, 0, 0);
		hWndC = NULL;
		while (capVideoYield(NULL) && GetTickCount() < t)
			;
	}
	hwVidParent = NULL;
}

int VideoDevs()
{
	return (hWndC != NULL);
}