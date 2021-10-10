//
//  FundamentalMat.hpp
//  StructureFromMotion
//
//  Created by Denisa on 5.10.21.
//

#ifndef FundamentalMat_hpp
#define FundamentalMat_hpp

#include <stdio.h>
#include "Includes.hpp"

class FundamentalMat
{
public: FundamentalMat(int iter = 0, int t = 0);
    cv::Matx33d FRansac(std::vector<cv::Point2d> points_src, std::vector<cv::Point2d> points_trg, std::vector<int> &inliersId);
    cv::Matx33d computeF(std::vector<cv::Point2d> points_src, std::vector<cv::Point2d> points_trg);
    std::vector<int> numInliners(cv::Matx33d F, std::vector<cv::Point2d> points_src, std::vector<cv::Point2d> points_trg);
    void testFundamentalMat();
private:
    int iterations;
    int threshold;
};

#endif /* FundamentalMat_hpp */
