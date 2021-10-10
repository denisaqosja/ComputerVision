//
//  main.cpp
//  PanoramaStitching
//
//  Created by Denisa on 21.9.21.
//

#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <filesystem>
#include <unistd.h>
#include <mach-o/dyld.h>
#include <limits.h>

#include "ImageData.h"
#include "Matching.hpp"
#include "Ransac.hpp"
#include "StitchImage.hpp"

using namespace std;
using namespace cv;

int main(int argc, const char * argv[])
{
    /* Load images */
    vector<string> imageNames = {"data/7.jpg",  "data/8.jpg",  "data/9.jpg",  "data/10.jpg", "data/11.jpg", "data/12.jpg", "data/13.jpg", "data/14.jpg", "data/15.jpg"};
    
    vector<ImageData> imagesData;
    string filePath = "/Users/user/Desktop/C++Coding/ComputerVision/PanoramaStitching/PanoramaStitching/";
    auto i = 0;
    for (auto file : imageNames)
    {
        ImageData img;
        img.fileName = file;
        img.image = cv::imread(filePath + file);
        resize(img.image, img.image, cv::Size(), 0.7, 0.7, cv::INTER_AREA);
        img.id = i++;
        assert(img.image.data);
        imagesData.push_back(img);
        cout << "Loaded Image " << img.id << " " << img.fileName << " of size " << img.image.cols << "x" << img.image.rows << endl;
    }
    
    /* Feature detection */
    for (ImageData& img : imagesData)
    {
        img.computeFeatures();
    }
    
    /* Feature matching */
    Matching objMatch;
    Ransac objRansac(2, 1000);
    
    for (int i = 1; i < imagesData.size(); i++)
    {
        //compute matches
        
        vector<DMatch> matches = objMatch.computeMatches(imagesData[i-1], imagesData[i]);
        
        {
             // Debug output
             cv::Mat3b matchImg = objMatch.createMatchImage(imagesData[i - 1], imagesData[i], matches);
             int h             = 200;
             int w             = (float(matchImg.cols) / matchImg.rows) * h;
             resize(matchImg, matchImg, Size(w, h));
             auto name = "Matches (" + std::to_string(i - 1) + "," + std::to_string(i) + ") " + imagesData[i - 1].fileName + " - " + imagesData[i].fileName;
            
             cv::imwrite(filePath + "output/matching" + std::to_string(imagesData[i-1].id) + ".png" , matchImg);
         }

         auto H = objRansac.computeHomographyRansac(imagesData[i - 1], imagesData[i], matches);
         imagesData[i].HtoPrev = H.inv();
         imagesData[i - 1].HtoNext = H;
        
    }
    // =========== Stiching ===========
    
    StitchImage objStitcher;
    cv::Mat3b stitchedImage = objStitcher.createStichedImage(imagesData);
    imwrite(filePath + "output/Panorama.png", stitchedImage);
    return 0;
}
