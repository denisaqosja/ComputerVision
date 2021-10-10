//
//  Ransac.cpp
//  PanoramaStitching
//
//  Created by Denisa on 3.10.21.
//
#include <cstdlib>
#include <random>
#include <vector>
#include <chrono>

#include "Ransac.hpp"

Ransac::Ransac(int t, int iter)
{
    threshold = t;
    iterations = iter;
}

int Ransac::numberInliers(std::vector<cv::Point2d> point1, std::vector<cv::Point2d> point2, cv::Matx33d H)
{
    int numInliers = 0;
    
    for(int i = 0; i < point1.size(); i++)
    {
        //create homogenoues points
        cv::Vec3d p, q, p_estimated;
        
        p = cv::Vec3d(point1[i].x, point1[i].y, 1);
        q = cv::Vec3d(point2[i].x, point2[i].y, 1);
        
        p_estimated = H * p;
        
        //distance of q with p_estimated;
        double distance = cv::norm(q, p_estimated);
        if(distance <= threshold)
            numInliers++;
    }
    return numInliers;
}


cv::Matx33d Ransac::computeHomographyRansac(ImageData& img1, ImageData& img2, std::vector<cv::DMatch>& matches)
{
    //obtain 2d points from detected and matched keypoints
    std::vector<cv::Point2d> src_points;
    std::vector<cv::Point2d> trg_points;
    
    for(int i = 0; i < matches.size(); i++)
    {
        //DMatch objects have the attributes: distance, queryIdx, trainIdx, imgIdx;
        //queryIdx: gives the index of the descriptor in the list of query descriptors (in our case, it’s the list of descriptors in the img1).
        //trainIdx: gives the index of the descriptor in the list of train descriptors (in our case, it’s the list of descriptors in the img2).
        src_points.push_back(img1.keypoints[matches[i].queryIdx].pt);
        trg_points.push_back(img2.keypoints[matches[i].trainIdx].pt);
    }
    
    int numPoints = 4;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution{0, (int)src_points.size() - 1};
    
    cv::Matx33d bestH, H;
    int bestNumInliers = 0;
    
    for (int iter = 0; iter < iterations; iter++)
    {
        //select 4 random points from matches
        std::vector<cv::Point2d> fourMatches_src;
        std::vector<cv::Point2d> fourMatches_trg;
            
        for(int value = 0; value < numPoints; value++)
        {
            int random_idx = distribution(gen);
            fourMatches_src.push_back(src_points[random_idx]);
            fourMatches_trg.push_back(trg_points[random_idx]);
        }
        //std::cout<<" Four matches src \n "<<fourMatches_src <<std::endl;
        //std::cout<<" Four matches trg \n "<<fourMatches_trg <<std::endl;
        
        //compute Homography
        H = cv::findHomography(fourMatches_src, fourMatches_trg);
        
        //Count the number of inliers with the specified H
        int numInliers = numberInliers(src_points, trg_points, H);
        if(numInliers > bestNumInliers)
        {
            bestNumInliers = numInliers;
            //save best H
            bestH = H;
        }
        
    }
    std::cout << "(" << img1.id << "," << img2.id << ") found " << bestNumInliers << " RANSAC inliers." << std::endl;
    
    return bestH;
}

