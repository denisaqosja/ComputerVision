//
//  ImageData.h
//  PanoramaStitching
//
//  Created by Denisa on 21.9.21.
//

#ifndef ImageData_h
#define ImageData_h

#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/viz.hpp>

struct ImageData
{
    std::string fileName;
    int id;
    cv::Mat3b image;

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat1b descriptors;

    // Init with identity matrix
    cv::Matx33d HtoReference = cv::Matx33d::eye();
    cv::Matx33d HtoPrev      = cv::Matx33d::eye();
    cv::Matx33d HtoNext      = cv::Matx33d::eye();

    void computeFeatures()
    {
        static thread_local cv::Ptr<cv::ORB> detector = cv::ORB::create(2000);
        detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
        std::cout << "Found " << keypoints.size() << " ORB features on image " << id << std::endl;
    }
};


#endif /* ImageData_h */
