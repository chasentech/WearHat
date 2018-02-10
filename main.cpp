#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void mythreshold(Mat &img, uchar T, bool flag)
{
	int n1 = img.rows;
	int nc = img.cols * img.channels();
	if (img.isContinuous())//判断图像是否连续
	{
		nc = nc * n1;
		n1 = 1;
	}
	for (int i = 0; i < n1; i++)
	{
		uchar *p = img.ptr<uchar>(i);
		for (int j = 0; j < nc; j++)
		{
			if (flag == true)
			{
				if (p[j] < T)
					p[j] = 0;
				else p[j] = 255;
			}
			if (flag == false)
			{
				if (p[j] > T)
					p[j] = 0;
				else p[j] = 255;
			}
		}
	}
}

void add_logo(Mat &img, Mat &logo, int thresh, Point start)
{
	//二值化制作掩码
	Mat logo_gray;
	cvtColor(logo, logo_gray, COLOR_BGR2GRAY);
	mythreshold(logo_gray, thresh, false);
	//imshow("gray", logo_gray);
	Mat mask = logo_gray;

	//与原图像融合的区域
	Rect r1(start.x, start.y, logo.cols, logo.rows);
	Mat img_show = img(r1);
	
	//通过掩码添加logo
	logo.copyTo(img_show, mask);
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	double scale, bool tryflip, Point &cen, int &rad)
{
	int i = 0;
	double t = 0;
	//建立用于存放人脸的向量容器
	vector<Rect> faces, faces2;
	//定义一些颜色，用来标示不同的人脸
	const static Scalar colors[] = { CV_RGB(0, 0, 255),
		CV_RGB(0, 128, 255),
		CV_RGB(0, 255, 255),
		CV_RGB(0, 255, 0),
		CV_RGB(255, 128, 0),
		CV_RGB(255, 255, 0),
		CV_RGB(255, 0, 0),
		CV_RGB(255, 0, 255) };
	//建立缩小的图片，加快检测速度
	//nt cvRound (double value) 对一个double型的数进行四舍五入，并返回一个整型数！
	Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);
	//转成灰度图像，Harr特征基于灰度图
	cvtColor(img, gray, CV_BGR2GRAY);
	//改变图像大小，使用双线性差值
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	//变换后的图像进行直方图均值化处理
	equalizeHist(smallImg, smallImg);

	//程序开始和结束插入此函数获取时间，经过计算求得算法执行时间
	t = (double)cvGetTickCount();
	//检测人脸
	//detectMultiScale函数中smallImg表示的是要检测的输入图像为smallImg，faces表示检测到的人脸目标序列，1.1表示
	//每次图像尺寸减小的比例为1.1，2表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大
	//小都可以检测到人脸),CV_HAAR_SCALE_IMAGE表示不是缩放分类器来检测，而是缩放图像，Size(30, 30)为目标的
	//最小最大尺寸
	cascade.detectMultiScale(smallImg, faces,
		1.1, 2, 0
		//|CV_HAAR_FIND_BIGGEST_OBJECT
		//|CV_HAAR_DO_ROUGH_SEARCH
		| CV_HAAR_SCALE_IMAGE
		,
		Size(30, 30));
	//如果使能，翻转图像继续检测
	if (tryflip)
	{
		/*flipCode，翻转模式，flipCode == 0垂直翻转（沿X轴翻转），flipCode>0水平翻转（沿Y轴翻转），
			flipCode<0水平垂直翻转（先沿X轴翻转，再沿Y轴翻转，等价于旋转180°）*/
		flip(smallImg, smallImg, 1);
		cascade.detectMultiScale(smallImg, faces2,
			1.1, 2, 0
			//|CV_HAAR_FIND_BIGGEST_OBJECT
			//|CV_HAAR_DO_ROUGH_SEARCH
			| CV_HAAR_SCALE_IMAGE
			,
			Size(30, 30));
		for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++)
		{
			faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
		}
	}
	t = (double)cvGetTickCount() - t;
	//   qDebug( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
	for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
	{
		Mat smallImgROI;
		vector<Rect> nestedObjects;
		Point center;
		Scalar color = colors[i % 8];
		int radius;

		double aspect_ratio = (double)r->width / r->height;
		if (0.75 < aspect_ratio && aspect_ratio < 1.3)
		{
			//标示人脸时在缩小之前的图像上标示，所以这里根据缩放比例换算回去
			center.x = cvRound((r->x + r->width*0.5)*scale);
			center.y = cvRound((r->y + r->height*0.5)*scale);
			radius = cvRound((r->width + r->height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);

			cen = center;
			rad = radius;

		}
		else
			rectangle(img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
			cvPoint(cvRound((r->x + r->width - 1)*scale), cvRound((r->y + r->height - 1)*scale)),
			color, 3, 8, 0);
		//if (nestedCascade.empty())
		//	continue;
		//smallImgROI = smallImg(*r);
		////同样方法检测人眼
		//nestedCascade.detectMultiScale(smallImgROI, nestedObjects,
		//	1.1, 2, 0
		//	//|CV_HAAR_FIND_BIGGEST_OBJECT
		//	//|CV_HAAR_DO_ROUGH_SEARCH
		//	//|CV_HAAR_DO_CANNY_PRUNING
		//	| CV_HAAR_SCALE_IMAGE
		//	,
		//	Size(30, 30));
		//for (vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++)
		//{
		//	cout << "eyes" << endl;
		//	center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
		//	center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
		//	radius = cvRound((nr->width + nr->height)*0.25*scale);
		//	circle(img, center, radius, color, 3, 8, 0);
		//}
	}
	imshow("result", img);
}

void output_text()
{
	//输出欢迎信息和OpenCV版本
	printf("\n\n\t\t\t趣味小项目之戴帽子\n");
	printf("\n\n\t\t\tOpenCV版本为：" CV_VERSION);
	printf("\n");
	printf("----------------------------------------------------------------------------\n");
	printf("操作步骤：\n");
	printf("\tSpace键：换帽子\n");
	printf("\tEsc键：  退出程序\n");
	printf("----------------------------------------------------------------------------\n");
}

/***************************测试阈值*********************************/
Mat test_hat = imread("E:/mygithub/WearHat/hat1.png", -1);
int test_threshold = 90;
void on_Trackbar(int, void*)
{
	Mat test_img = imread("E:/mygithub/WearHat/img.jpg", 1);
	//二值化制作掩码
	Mat logo_gray;
	cvtColor(test_hat, logo_gray, COLOR_BGR2GRAY);
	mythreshold(logo_gray, test_threshold, false);
	imshow("gray", logo_gray);
	Mat mask = logo_gray;
	imwrite("hat1_mask.jpg", logo_gray);

	//与原图像融合的区域
	Rect r1(50, 50, logo_gray.cols, logo_gray.rows);
	Mat img_show = test_img(r1);

	//通过掩码添加logo
	test_hat.copyTo(img_show, mask);
	imshow("添加logo", test_img);

	cout << "轨迹条位置：" << getTrackbarPos("阈值", "test") << endl;
}
/***************************测试阈值*********************************/

int main(int argc, char **argv)
{
	output_text();
	/***************************测试阈值*********************************/
	//轨迹条
	//resize(test_hat, test_hat, Size(400, 400));
	//imshow("test_hat", test_hat);
	//imwrite("hat_3.png", test_hat);

	//hat1: 90
	//hat2: 62
	//hat3: 246

	//namedWindow("test", 0);
	//createTrackbar("阈值", "test", &test_threshold, 255, on_Trackbar);
	//on_Trackbar(test_threshold, 0);
	/***************************测试阈值*********************************/

	//检测人脸
	CascadeClassifier cascade, nestedCascade;
	//训练好的文件名称，放置在可执行文件同目录下
	cascade.load("E:/mygithub/WearHat/haarcascade_frontalface_alt.xml");


	int hat_number = 1;		//帽子类型
	Mat img;
	VideoCapture cap(0);
	cout << "帽子类型为: " << 1 << endl;
	while (1)
	{
		cap >> img;
		if (img.empty()) break;
		//imshow("img", img);

		Point center;
		int radius = 0;
		Mat img_dete;
		img.copyTo(img_dete);
		detectAndDraw(img_dete, cascade, 2, 0, center, radius);
		//imshow("img_dete", img_dete);
		//cout << "圆心:" << center << " ,半径:" << radius << endl;

		Mat img_temp;
		if (center.x != 0 && radius != 0)
		{
			Mat hat;//logo矩阵
			int my_thresh;		//阈值
			Point add_point = Point(0, 0);	//添加logo起始点
			switch (hat_number)
			{
			case 1: hat = imread("E:/mygithub/WearHat/hat1.png", -1); //载入带Alpha图像
				my_thresh = 90;
				resize(hat, hat, Size(hat.cols * radius / 180, hat.rows * radius / 180));
				add_point = Point(center.x - (hat.cols >> 1), center.y - radius - (50 + 130 * radius / 180)); //起始点
				if (add_point.x < 0 || add_point.y < 0)
					add_point = Point(0, 0);
				break;
			case 2: hat = imread("E:/mygithub/WearHat/hat2.png", -1); //载入带Alpha图像
				my_thresh = 62;
				resize(hat, hat, Size(hat.cols * radius / 200, hat.rows * radius / 200));
				add_point = Point(center.x - (hat.cols >> 1) - 40, center.y - radius - (40 + 130 * radius / 200)); //起始点
				if (add_point.x < 0 || add_point.y < 0)
					add_point = Point(0, 0);
				break;
			case 3: hat = imread("E:/mygithub/WearHat/hat3.png", -1); //载入带Alpha图像
				my_thresh = 246;
				resize(hat, hat, Size(hat.cols * radius / 170, hat.rows * radius / 170));
				add_point = Point(center.x - (hat.cols >> 1) + 15, center.y - radius - (130 + 130 * radius / 190)); //起始点
				if (add_point.x < 0 || add_point.y < 0)
					add_point = Point(0, 0);
				break;
			case 4: cout << "敬请期待" << endl;
				break;
			}

			if (hat.empty())
				continue;
			//imshow("hat", hat);

			//添加logo
			if (add_point.x != 0 && add_point.y != 0)
			{
				img.copyTo(img_temp);
				add_logo(img_temp, hat, my_thresh, add_point);
			}
			else img_temp = img;
		}
		else 
			img_temp = img;

		imshow("添加logo", img_temp);

		char key = waitKey(30);
		switch (key)
		{
			case 27: return 0; break;
			case 32: hat_number++;
				if (hat_number == 4)
				{
					hat_number = 1;
					cout << "帽子类型为: " << hat_number << endl;
				}
				else
					cout << "帽子类型为: " << hat_number << endl;

				break;
		}
	}

	//载入源图

	//img = imread("E:/mygithub/WearHat/img.jpg", 1); //载入源图像
	//if (img.empty()) return -1;
	//imshow("img", img);






	//waitKey(0);
	return 0;
	
}