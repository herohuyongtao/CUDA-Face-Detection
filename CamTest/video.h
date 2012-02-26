// video.h

void StartVideoCapture(HWND hwShow, int xpos, int ypos);
void StopVideoCapture();

void StartVideo(HWND hwParent, int visible);
void StopVideo();
void PollVideo();


int VideoDevs();
void ClearWindowArea(HWND hWnd, int windowNumber);