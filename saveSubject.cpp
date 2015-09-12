#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;

void help()
{
	cout << "\nThis program demonstrates the cascade recognizer. Now you can use Haar or LBP features.\n"
		"This classifier can recognize many ~rigid objects, it's most known use is for faces.\n"
		"Usage:\n"
		"./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
		"   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
		"   [--scale=<image scale greater or equal to 1, try 1.3 for example>\n"
		"   [filename|camera_index]\n\n"
		"see facedetect.cmd for one call:\n"
		"./facedetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"../../data/haarcascades/haarcascade_eye.xml\" --scale=1.3 \n"
		"Hit any key to quit.\n"
		"Using OpenCV version " << CV_VERSION << "\n" << endl;
}

void detectAndDraw( Mat& img,
		CascadeClassifier& cascade, CascadeClassifier& nestedCascade,
		double scale,char* argv[]);

String cascadeName = "/Users/dhwanit/Desktop/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_alt.xml";
String nestedCascadeName = "/Users/dhwanit/Desktop/opencv-2.4.9/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
int count_face = 0;


int main( int argc, char* argv[] )
{
	CvCapture* capture = 0;
	Mat frame, frameCopy, image;
	const String scaleOpt = "--scale=";
	size_t scaleOptLen = scaleOpt.length();
	const String cascadeOpt = "--cascade=";
	size_t cascadeOptLen = cascadeOpt.length();
	const String nestedCascadeOpt = "--nested-cascade";
	size_t nestedCascadeOptLen = nestedCascadeOpt.length();
	String inputName;
	//	help();

	CascadeClassifier cascade, nestedCascade;
	double scale = 1;

	for( int i = 1; i < argc; i++ )
	{
		cout << "Processing " << i << " " <<  argv[i] << endl;
		if( cascadeOpt.compare( 0, cascadeOptLen, argv[i], cascadeOptLen ) == 0 )
		{
			cascadeName.assign( argv[i] + cascadeOptLen );
			cout << "  from which we have cascadeName= " << cascadeName << endl;
		}
		else if( nestedCascadeOpt.compare( 0, nestedCascadeOptLen, argv[i], nestedCascadeOptLen ) == 0 )
		{
			if( argv[i][nestedCascadeOpt.length()] == '=' )
				nestedCascadeName.assign( argv[i] + nestedCascadeOpt.length() + 1 );
			if( !nestedCascade.load( nestedCascadeName ) )
				cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
		}
		else if( scaleOpt.compare( 0, scaleOptLen, argv[i], scaleOptLen ) == 0 )
		{
			if( !sscanf( argv[i] + scaleOpt.length(), "%lf", &scale ) || scale < 1 )
				scale = 1;
			cout << " from which we read scale = " << scale << endl;
		}
		else if( argv[i][0] == '-' )
		{
			cerr << "WARNING: Unknown option %s" << argv[i] << endl;
		}
		else
			inputName.assign( argv[i] );
	}

	if( !cascade.load( cascadeName ) )
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		cerr << "Usage: facedetect [--cascade=<cascade_path>]\n"
			"   [--nested-cascade[=nested_cascade_path]]\n"
			"   [--scale[=<image scale>\n"
			"   [filename|camera_index]\n" << endl ;
		return -1;
	}

	capture = cvCaptureFromCAM( 0 );
	int c = inputName.empty() ? 0 : inputName.c_str()[0] - '0' ;
	if(!capture) cout << "Capture from CAM " <<  c << " didn't work" << endl;

	cvNamedWindow( "result", 1 );

	if( capture )
	{
		cout << "In capture ..." << endl;
		for(;;)
		{
			IplImage* iplImg = cvQueryFrame( capture );
			frame = iplImg;
			if( frame.empty() )
				break;
			if( iplImg->origin == IPL_ORIGIN_TL )
				frame.copyTo( frameCopy );
			else
				flip( frame, frameCopy, 0 );

			detectAndDraw( frameCopy, cascade, nestedCascade, scale, argv );

			if( waitKey( 10 ) >= 0 )
				goto _cleanup_;
		}

		waitKey(0);

_cleanup_:
		cvReleaseCapture( &capture );
	}

	cvDestroyWindow("result");

	return 0;
}

void detectAndDraw( Mat& img,
		CascadeClassifier& cascade, CascadeClassifier& nestedCascade,
		double scale, char* argv[])
{
	int i = 0;
	double t = 0;
	vector<Rect> faces;
	const static Scalar colors[] =  { CV_RGB(0,0,255),
		CV_RGB(0,128,255),
		CV_RGB(0,255,255),
		CV_RGB(0,255,0),
		CV_RGB(255,128,0),
		CV_RGB(255,255,0),
		CV_RGB(255,0,0),
		CV_RGB(255,0,255)} ;
	Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

	cvtColor( img, gray, CV_BGR2GRAY );

	resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
	equalizeHist( smallImg, smallImg );

	t = (double)cvGetTickCount();
	cascade.detectMultiScale( smallImg, faces,
			1.1, 2, 0
			//|CV_HAAR_FIND_BIGGEST_OBJECT
			//|CV_HAAR_DO_ROUGH_SEARCH
			|CV_HAAR_SCALE_IMAGE
			,
			Size(30, 30) );
	t = (double)cvGetTickCount() - t;
	for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
	{
		Mat smallImgROI;
		vector<Rect> nestedObjects;
		Point center;
		Scalar color = colors[i%8];
		int radius;
		center.x = cvRound((r->x + r->width*0.5)*scale);
		center.y = cvRound((r->y + r->height*0.5)*scale);
		radius = cvRound((r->width + r->height)*0.25*scale);
		Rect rect = Rect(center.x - radius , center.y - radius , 2*radius , 2*radius );
		Mat faceImage = gray(rect);
		char s1[100];
		sprintf(s1,"dataset/%s/%s_%d.jpg",argv[1],argv[1],count_face++);
		imwrite(s1,faceImage);
		//imshow( "result", faceImage );
		//circle( img, center, radius, color, 3, 8, 0 );
		rectangle(img, rect, Scalar(255,0,0));
		if( nestedCascade.empty() )
			continue;
		smallImgROI = smallImg(*r);
		nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
				1.1, 2, 0
				//|CV_HAAR_FIND_BIGGEST_OBJECT
				//|CV_HAAR_DO_ROUGH_SEARCH
				//|CV_HAAR_DO_CANNY_PRUNING
				|CV_HAAR_SCALE_IMAGE
				,
				Size(30, 30) );
		for( vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++ )
		{
			center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
			center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
			radius = cvRound((nr->width + nr->height)*0.25*scale);
			circle( img, center, radius, color, 3, 8, 0 );
			Rect rect1 = Rect(center.x - radius , center.y - radius , 2*radius , 2*radius );
			Mat faceImage1 = gray(rect1);
			char s2[100];
			sprintf(s2,"dataset/%s/%s_%d.jpg",argv[1],argv[1],count_face++);
			imwrite(s2,faceImage1);
		}
	}
	//cv::imshow( "result", img );
}
