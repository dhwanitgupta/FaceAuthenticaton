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

#include <stdio.h>      /* printf */
#include <stdlib.h>
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
		double scale);

String cascadeName = "/Users/dhwanit/Desktop/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_alt.xml";
String nestedCascadeName = "/Users/dhwanit/Desktop/opencv-2.4.9/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
Ptr<FaceRecognizer> model;
int faceCounter = 1;
int stdHeight = 120;
int stdWidth = 120;
void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file)
		throw std::exception();
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		Mat imageRead = imread(path,0);
		resize(imageRead, imageRead, Size(stdHeight,stdWidth), 0, 0, INTER_CUBIC);
		images.push_back(imageRead);
		labels.push_back(atoi(classlabel.c_str()));
	}
}

void trainFaces(){
	string fn_csv = string("data.csv");
	vector<Mat> images;
	vector<int> labels;
	try {
		read_csv(fn_csv, images, labels);
	} catch (exception&) {
		cerr << "Error opening file \"" << fn_csv << "\"." << endl;
		exit(1);
	}

	model = createLBPHFaceRecognizer();
	model->train(images, labels);
	cout << "Model Information:" << endl;
	string model_info = format("\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
			model->getInt("radius"),
			model->getInt("neighbors"),
			model->getInt("grid_x"),
			model->getInt("grid_y"),
			model->getDouble("threshold"));
	cout << model_info << endl;
	// We could get the histograms for example:
	vector<Mat> histograms = model->getMatVector("histograms");
	//         // But should I really visualize it? Probably the length is interesting:
	cout << "Size of the histograms: " << histograms[0].total() << endl;
}

int main( int argc, const char** argv )
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
	trainFaces();
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

	capture = cvCaptureFromCAM(1); 
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

			detectAndDraw( frameCopy, cascade, nestedCascade, scale );

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

void openDoor(Mat img){
	system("python aurdino.py");
	
}
void testKnownFace(Mat img){
	char saveImage[100];
	sprintf(saveImage, "FacesFound/resultImage%d.jpg", faceCounter++);
	imwrite(saveImage,img);
	resize(img, img, Size(stdHeight,stdWidth), 0, 0, INTER_CUBIC);
	int predictedLabel = -1;
	double confidence = 0.0;
	model->predict(img, predictedLabel, confidence);
	if( !( predictedLabel == 1 || predictedLabel == 5 || predictedLabel == 2 ) && confidence < 80 ) {
		cout << "ImageName " << saveImage << "  Label = " << predictedLabel << " Confidence = " << confidence << endl;
		openDoor(img);
	}
	//Mat eigenvalues = model->getMat("eigenvalues");
	// And we can do the same to display the Eigenvectors (read Eigenfaces):
	//Mat W = model->getMat("eigenvectors");
	//         // Get the sample mean from the training data
	//Mat mean = model->getMat("mean");

	//imshow("MeanImage" ,mean);
	//cout << "W size "  << W.size() << endl;
	/*for (int i = 0; i < min(10, W.cols); i++) {
		string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
		cout << msg << endl;
		// get eigenvector #i
		Mat ev = W.col(i).clone();
		//                 // Reshape to original size & normalize to [0...255] for imshow.
		Mat grayscale =  ev ; // norm_0_255(ev.reshape(1, height));
		//                                 // Show the image & apply a Jet colormap for better sensing.
		Mat cgrayscale;
		applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
		// Display or save:
		imshow(format("eigenface_%d", i), cgrayscale);

	}*/
}
void detectAndDraw( Mat& img,
		CascadeClassifier& cascade, CascadeClassifier& nestedCascade,
		double scale)
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
			1.1, 12, 0
			|CV_HAAR_FIND_BIGGEST_OBJECT
			//|CV_HAAR_DO_ROUGH_SEARCH
			//|CV_HAAR_SCALE_IMAGE
			,
			Size(60, 60) );
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
		testKnownFace(faceImage);
		//imshow( "result", faceImage );
		//circle( img, center, radius, color, 3, 8, 0 );
		rectangle(img, rect, Scalar(255,0,0));
		//if( nestedCascade.empty() )
			continue;
		smallImgROI = smallImg(*r);
		nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
				1.1, 12, 0
				|CV_HAAR_FIND_BIGGEST_OBJECT
				//|CV_HAAR_DO_ROUGH_SEARCH
				//|CV_HAAR_DO_CANNY_PRUNING
				//|CV_HAAR_SCALE_IMAGE
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
			testKnownFace(faceImage1);
		}
	}
	cv::imshow( "result", img );
}
