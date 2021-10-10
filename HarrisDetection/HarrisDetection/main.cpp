//
//  main.cpp
//  HarrisDetection
//
//  Created by Denisa on 20.9.21.
//

#include "Harris.hpp"

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    
    Harris harrisObj("/Users/user/Desktop/C++Coding/ComputerVision/HarrisDetection/HarrisDetection/", "blox.jpg", -0.01, 0.1);
    cv::Mat1b grayImage = harrisObj.loadImage();
    
    //1. Convert image into float grayscale in the range [0,1]
    cv::Mat1f image;
    grayImage.convertTo(image, image.type());
    image *= (1.0f / 255.0f);
    
    //2. Harris response
    Harris::Derivatives d;
    cv::Mat1f response;
    d = harrisObj.imageDerivatives(image);
    response = harrisObj.harrisResponse(image, d);
    
    //3. Draw corners and edges
    auto points = harrisObj.harrisKeypoints(response);
    
    cv::Mat3b rgbImage, keypointsImage, edges;
    cvtColor(grayImage, rgbImage, COLOR_GRAY2RGB);
    
    drawKeypoints(rgbImage, points, keypointsImage, Scalar(0, 255, 0));
    edges = harrisObj.harrisEdges(rgbImage, response);

    //4. Write results
    harrisObj.writeRes(d, response, keypointsImage, edges);
    
    return 0;
}
