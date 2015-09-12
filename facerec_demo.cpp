
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

Mat toGrayscale(InputArray _src) {
    Mat src = _src.getMat();
    // only allow one channel
    if(src.channels() != 1)
        CV_Error(CV_StsBadArg, "Only Matrices with one channel are supported");
    // create and return normalized image
    Mat dst;
    cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}

void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file)
        throw std::exception();
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        images.push_back(imread(path, 0));
        labels.push_back(atoi(classlabel.c_str()));
    }
}

int main(int argc, const char *argv[]) {
    // check for command line arguments
    if (argc != 2) {
        cout << "usage: " << argv[0] << " <csv.ext>" << endl;
        exit(1);
    }
    // path to your CSV
    string fn_csv = string(argv[1]);
    // images and corresponding labels
    vector<Mat> images;
    vector<int> labels;
    // read in the data
    try {
        read_csv(fn_csv, images, labels);
    } catch (exception&) {
        cerr << "Error opening file \"" << fn_csv << "\"." << endl;
        exit(1);
    }
    // get width and height
    //int width = images[0].cols;
    int height = images[0].rows;
    // get test instances
    string test_image;
    cin >> test_image;
    Mat testSample = imread(test_image,0);
   // Mat testSample;
   // cvtColor(testSample1, testSample, CV_BGR2GRAY);
    int testLabel = 0;

    cout << images.size() << " " <<testSample.rows << " " << testSample.cols << endl;
    // ... and delete last element
    //images.pop_back();
    //labels.pop_back();
    // build the Fisherfaces model
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    model->train(images, labels);
    // test model
    int predictedLabel = -1;
    double confidence = 0.0;
    predictedLabel = model->predict(testSample);
    cout << "predicted class = " << predictedLabel << endl;
    cout << "actual class = " << testLabel << endl;
   // cout << "confidence = " << confidence << endl;
    
    // get the eigenvectors
  /*  Mat W = model->eigenvectors();
    // show first 10 fisherfaces
    for (int i = 0; i < min(10, W.cols); i++) {
        // get eigenvector #i
        Mat ev = W.col(i).clone();
        // reshape to original size AND normalize between [0...255]
        Mat grayscale = toGrayscale(ev.reshape(1, height));
        // show image (with Jet colormap)
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
        imshow(format("%d", i), cgrayscale);
    }
    waitKey(0);*/
    return 0;
}
