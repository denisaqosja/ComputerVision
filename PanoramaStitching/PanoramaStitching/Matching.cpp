//
//  Matching.cpp
//  PanoramaStitching
//
//  Created by Denisa on 21.9.21.
//

#include <vector>

#include "Matching.hpp"
#include "ImageData.h"

Matching::Matching(){}

std::vector<std::vector<cv::DMatch> > Matching::matchKNN(cv::Mat1b& descriptor1, cv::Mat1b& descriptor2)
{
    // Initialize N x 2 array of matches.
    std::vector<std::vector<cv::DMatch> > knnMatches(descriptor1.rows, std::vector<cv::DMatch>(2, cv::DMatch(-1, -1, 1000)));
    //find 2 neareast descriptors in image 2 for each descriptor in image 1
    double distance;
    for(int i = 0; i<descriptor1.rows; i++)
        for(int j = 0; j < descriptor2.rows; j++){
            distance = norm(descriptor1.row(i), descriptor2.row(j), cv::NORM_HAMMING);
            
            cv::DMatch dm(i, j, distance);
            if(dm.distance < knnMatches[i][0].distance){
                knnMatches[i][1] = knnMatches[i][0];
                knnMatches[i][0] = dm;
            }
            else if(dm.distance < knnMatches[i][1].distance){
                knnMatches[i][1] = dm;
            }
        }
    return knnMatches;
}


std::vector<cv::DMatch> Matching::ratioTest(std::vector<std::vector<cv::DMatch> > knnMatches, float threshold)
{
    std::vector<cv::DMatch>matches;
    
    for (int i = 0; i < knnMatches.size(); i++)
    {
        double ratio = knnMatches[i][0].distance / knnMatches[i][1].distance;
        if (abs(ratio) < threshold)
            matches.push_back(knnMatches[i][0]);
    }
   
    return matches;
}



std::vector<cv::DMatch> Matching::computeMatches(ImageData& img1, ImageData& img2)
{
    auto knnMatches = matchKNN(img1.descriptors, img2.descriptors);
    auto matches = ratioTest(knnMatches, 0.7);
    std::cout<<"(" << img1.id << "," << img2.id << ") found " << matches.size() << " matches." << std::endl;
    
    return matches;
}

