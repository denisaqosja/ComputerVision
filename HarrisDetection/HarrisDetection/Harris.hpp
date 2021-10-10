//
//  Harris.hpp
//  HarrisDetection
//
//  Created by Denisa on 20.9.21.
//

#ifndef Harris_hpp
#define Harris_hpp

#include <stdio.h>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

class Harris
{
public:
    Harris(std::string path = "", std::string name = "", float edgeT = 0, float t = 0);
    
    struct Derivatives{
        cv::Mat1f A;
        cv::Mat1f B;
        cv::Mat1f C;
    };
    
    cv::Mat1b loadImage();
    Derivatives imageDerivatives(cv::Mat1f grayImage);
    cv::Mat1f harrisResponse(cv::Mat1f image, Derivatives d);
    
    std::vector<KeyPoint> harrisKeypoints(cv::Mat1f harrisResponse);
    cv::Mat3b harrisEdges(cv::Mat3b image, cv::Mat1f R);
    void writeRes(Derivatives d, cv::Mat1f response, cv::Mat3b keypointsImage, cv::Mat3b edges);
    
private:
    string dirPath;
    string fileName;
    float edgeThreshold;
    float threshold;
};


#endif /* Harris_hpp */
