//
//  FeatureMatching.hpp
//  StructureFromMotion
//
//  Created by Denisa on 5.10.21.
//

#ifndef FeatureMatching_hpp
#define FeatureMatching_hpp

#include <stdio.h>
#include "Includes.hpp"

class FeatureMatching
{
public: FeatureMatching();
    void matchFeature(cv::Mat3b image1, cv::Mat3b image2, std::vector<cv::Point2d> &points1, std::vector<cv::Point2d> &points2);
    
};

#endif /* FeatureMatching_hpp */
