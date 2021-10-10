//
//  Matching.hpp
//  PanoramaStitching
//
//  Created by Denisa on 21.9.21.
//

#ifndef Matching_hpp
#define Matching_hpp

#include <stdio.h>
#include <vector>
#include "ImageData.h"

class Matching
{
public:
    Matching();
    std::vector<cv::DMatch> computeMatches(ImageData& image1, ImageData& image2);
    std::vector<std::vector<cv::DMatch> > matchKNN(cv::Mat1b& descriptor1, cv::Mat1b& descriptor2);
    std::vector<cv::DMatch> ratioTest(std::vector<std::vector<cv::DMatch> > knnMatches, float threshold);
    
    inline cv::Mat3b createMatchImage(ImageData& img1, ImageData& img2, std::vector<cv::DMatch>& matches)
    {
        cv::Mat3b img_matches;
        cv::drawMatches(img1.image, img1.keypoints, img2.image, img2.keypoints, matches, img_matches, cv::Scalar(0, 255, 0),
                    cv::Scalar(0, 255, 0), std::vector<char>(), cv::DrawMatchesFlags::DEFAULT);
        return img_matches;
    }
    
};



#endif /* Matching_hpp */
