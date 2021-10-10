//
//  Ransac.hpp
//  PanoramaStitching
//
//  Created by Denisa on 3.10.21.
//

#ifndef Ransac_hpp
#define Ransac_hpp

#include <stdio.h>
#include "ImageData.h"

class Ransac
{
public:
    Ransac(int t = 2, int iter = 1000);
    int numberInliers(std::vector<cv::Point2d> point1, std::vector<cv::Point2d> point2, cv::Matx33d H);
    cv::Matx33d computeHomographyRansac(ImageData& img1, ImageData& img2, std::vector<cv::DMatch>& matches);
    
private:
    float threshold;
    int iterations;
};

#endif /* Ransac_hpp */
