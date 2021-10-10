//
//  Harris.cpp
//  HarrisDetection
//
//  Created by Denisa on 20.9.21.
//

#include "Harris.hpp"

using namespace std;
using namespace cv;

Harris::Harris(std::string path, std::string name, float edgeT, float t)
{
    dirPath = path;
    fileName = name;
    edgeThreshold = edgeT;
    threshold = t;
}

cv::Mat1b Harris::loadImage()
{
    string filePath = dirPath + fileName;
    cv::Mat3b rgbImage = imread(filePath);
    
    if(rgbImage.empty())
    {
        std::cout<<"Image is not found. \n";
    }
    
    cv::Mat1b grayImage = cv::Mat1b::zeros(rgbImage.size());
    cvtColor(rgbImage, grayImage, COLOR_BGR2GRAY);
  
    return grayImage;
}
Harris::Derivatives Harris::imageDerivatives(cv::Mat1f image)
{
    //Sobel gradient
    
    cv::Mat1f sobelX;
    cv::Mat1f sobelY;
    
    Sobel(image, sobelX, CV_32F, 1, 0, 3);
    Sobel(image, sobelY, CV_32F, 0, 1, 3);
    
    //compute second derivatives
    
    cv::Mat1f sobelXX, sobelYY, sobelXY;
    sobelXX = sobelX.mul(sobelX);
    sobelXY = sobelX.mul(sobelY);
    sobelYY = sobelY.mul(sobelY);
    
    cv::Mat1f A, B, C;
    GaussianBlur(sobelXX, A, cv::Size(3,3), 1, 0, BORDER_DEFAULT);
    GaussianBlur(sobelYY, B, cv::Size(3,3), 1, 0, BORDER_DEFAULT);
    GaussianBlur(sobelXY, C, cv::Size(3,3), 1, 0, BORDER_DEFAULT);
    
    Derivatives d;
    d.A = A;
    d.B = B;
    d.C = C;

    return d;
}

cv::Mat1f Harris::harrisResponse(cv::Mat1f image, Derivatives d)
{
    //Harris Response: R = Det - k * Trace*Trace
    
    float k = 0.06;
    cv::Mat1f det = d.A.mul(d.B) - d.C.mul(d.C);
    cv::Mat1f trace = d.A + d.B;
    cv::Mat1f harrisResponse = det - k * (trace.mul(trace));
    
    return harrisResponse;
}

std::vector<KeyPoint> Harris::harrisKeypoints(cv::Mat1f R)
{
    std::vector<KeyPoint> points;
    
    for(int x = 1; x < R.rows - 1; x++)
            for(int y = 1; y < R.cols - 1; y++)
    
                if(R(y,x) > threshold)
                 {
                    if(R(y,x)>R(y-1,x) && R(y,x)>R(y-1,x-1) && R(y,x)>R(y-1,x+1) && R(y,x)>R(y,x-1)
                      && R(y,x)>R(y+1,x-1) && R(y,x)>R(y+1,x) && R(y,x)>R(y+1,x+1) && R(y,x)>R(y,x+1))
                    {
                            KeyPoint kp (x,y,1);
                            points.push_back(kp);
                    }
                }
    
    return points;
}

cv::Mat3b Harris::harrisEdges(cv::Mat3b image, cv::Mat1f R)
{
    cv::Mat3b result = image.clone();
    
    for(int x = 1; x < R.rows-1; x++)
        for(int y = 1; y < R.cols-1; y++)
            if(R(x,y) < edgeThreshold )
            {
               if ((R(x,y)<R(x-1,y) && R(x,y)<R(x+1, y)) || (R(x,y)<R(x, y-1) && R(x,y)<R(x, y+1)))
                    result(x,y) = Vec3b(0, 0, 255);
            }

       return result;
}

void Harris::writeRes(Derivatives d, cv::Mat1f response, cv::Mat3b keypointsImage, cv::Mat3b edges)
{
    //derivatives
    cv::imwrite(dirPath + ("Results/DerivativeXX.png"), 255 * d.A);
    cv::imwrite(dirPath + ("Results/DerivativeYY.png"), 255 * d.B);
    cv::imwrite(dirPath + ("Results/DerivativeXY.png"), 255 * d.C);
    
    //normalize the response
    double min, max;
    cv::Mat1f normalizedResponse;
    minMaxLoc(response, &min, &max);
    response *= 1.0/max;
    normalize(response, normalizedResponse, 0, 1, NORM_MINMAX, CV_32FC1);
    cv::imwrite(dirPath + ("Results/response.png"), normalizedResponse * 255);
    
    //image with corners
    cv::imwrite(dirPath + ("Results/corners.png"), keypointsImage);
    
    //image with edges
    cv::imwrite(dirPath + ("Results/edges.png"), edges);
    
}
