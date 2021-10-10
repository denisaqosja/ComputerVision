//
//  Triangulate.hpp
//  StructureFromMotion
//
//  Created by Denisa on 5.10.21.
//

#ifndef Triangulate_hpp
#define Triangulate_hpp

#include <stdio.h>
#include <iostream>

#include "Includes.hpp"

class Triangulate
{
public: Triangulate();
    void testTriangulate();
    cv::Point3f triangulatePoints(cv::Matx34d P1, cv::Matx34d P2, cv::Point2d p1, cv::Point2d p2);
    
    std::vector<cv::Point3f> triangulate(cv::Matx34d P1, cv::Matx34d P2, cv::Matx33d K, std::vector<cv::Point2d> p1, std::vector<cv::Point2d> p2);
    
};

#endif /* Triangulate_hpp */
