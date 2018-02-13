#ifndef _PUTTEXT_H_
#define _PUTTEXT_H_

#include <windows.h>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;

extern void GetStringSize(HDC hDC, const char* str, int* w, int* h);
extern void putTextZH(Mat &dst, const char* str, Point org, Scalar color, int fontSize,
	const char *fn = "Arial", bool italic = false, bool underline = false);

#endif // PUTTEXT_H_