//
//  FeatureMatching.cpp
//  StructureFromMotion
//
//  Created by Denisa on 5.10.21.
//

#include "FeatureMatching.hpp"

using namespace std;
using namespace cv;

FeatureMatching::FeatureMatching(){}

void FeatureMatching::matchFeature(cv::Mat3b image1, cv::Mat3b image2, std::vector<cv::Point2d> &points1, std::vector<cv::Point2d> &points2)
{
    vector<KeyPoint> keypoints1, keypoints2;
    cv::Mat1b descriptor1, descriptor2;
    
    static thread_local cv::Ptr<cv::ORB> detector = cv::ORB::create(20000);
    detector->detectAndCompute(image1, cv::noArray(), keypoints1, descriptor1);
    detector->detectAndCompute(image2, cv::noArray(), keypoints2, descriptor2);
    
    cout<<"Detected features: "<<keypoints1.size()<<" "<<keypoints2.size()<<endl;
    
    //Matching based on FLANN
    cv::FlannBasedMatcher flannMatcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    
    vector<vector<DMatch>> matches;
    vector<DMatch> goodMatches;
    //knnMatch: find 2 best matches, NN1 and NN2
    flannMatcher.knnMatch(descriptor1, descriptor2, matches, 2);
    
    //ratio test
    for (auto & match : matches){
        if(match[0].distance<match[1].distance * 0.8)
            goodMatches.push_back(match[0]);
    }
    
    cout<<"Matches after ratio test: "<<goodMatches.size()<<endl;
    
    //draw matches
    
    //convert keypoints into point2d
    for(auto m : goodMatches)
    {
        points1.push_back(keypoints1[m.queryIdx].pt);
        points2.push_back(keypoints2[m.trainIdx].pt);
    }
}
