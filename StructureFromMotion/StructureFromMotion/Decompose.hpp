//
//  Decompose.hpp
//  StructureFromMotion
//
//  Created by Denisa on 8.10.21.
//

#ifndef Decompose_hpp
#define Decompose_hpp

#include <stdio.h>
#include "Includes.hpp"

class Decompose
{
public: Decompose();
    void decompose(cv::Matx33d E, cv::Matx33d &R1, cv::Matx33d &R2, cv::Vec3d &T1, cv::Vec3d &T2);
    cv::Matx34d relativeTransformation(cv::Matx33d E, cv::Matx33d K, std::vector<cv::Point2d> points1, std::vector<cv::Point2d> points2);
    
};
#endif /* Decompose_hpp */
