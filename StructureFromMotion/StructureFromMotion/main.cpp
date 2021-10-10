//
//  main.cpp
//  StructureFromMotion
//
//  Created by Denisa on 4.10.21.
//

#include <iostream>

#include "Includes.hpp"
#include "FeatureMatching.hpp"
#include "FundamentalMat.hpp"
#include "Decompose.hpp"
#include "Triangulate.hpp"

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    
    string dirPath = "/Users/user/Desktop/C++Coding/ComputerVision/StructureFromMotion/StructureFromMotion/";
    
    cv::Mat3b image1 = imread(dirPath + "data/img1.jpg");
    cv::Mat3b image2 = imread(dirPath + "data/img2.jpg");
    
    assert(image1.data);
    assert(image2.data);
    
    std:cout<<"Loaded images \n";
    
    //intrinsic matrix
    cv::Matx33d K = {2890, 0, 1440, 0, 2890, 960, 0, 0, 1};
    
    //--------------Feature Matching--------------------//
    
    FeatureMatching matching;
    vector<Point2d> points_src, points_trg;
    
    matching.matchFeature(image1, image2, points_src, points_trg);
    
    //--------Essential and Fundamental Matrices--------//
    
    FundamentalMat Fmatrix(10000, 4);
    Fmatrix.testFundamentalMat();
    std::vector<int> inliers;
    cv::Matx33d F = Fmatrix.FRansac(points_src, points_trg, inliers);
    
    cv::Matx33d E = K.t() * F * K;
    E *= 1.0/E(2,2);
    
    std::vector<cv::Point2d> inliers_points_src, inliers_points_trg;
    for (auto &i: inliers)
    {
        inliers_points_src.push_back(points_src[i]);
        inliers_points_trg.push_back(points_trg[i]);
    }
    
    //-------Compute relative transformation------------//
    //-------Computes the best rigid transformation between two images
    //       given the essential matrix E and some points matches.
    Decompose dec;
    
    cv::Matx34d View1 = cv::Matx34d::eye();
    cv::Matx34d View2 = dec.relativeTransformation(E, K, inliers_points_src, inliers_points_trg);
    
    //-------Triangulate inlier matches----------------//
    Triangulate T;
    T.testTriangulate();
    std::vector<cv::Point3f> wps = T.triangulate(View1, View2, K, inliers_points_src, inliers_points_trg);
    
    wps.erase(std::remove_if(wps.begin(), wps.end(), [](Point3f p) { return p.z > 10 || p.y > 1; }), wps.end());
    
    //-----------------Rendering-----------------------//

    vector<Vec3b> colors(wps.size(), Vec3b(255, 0, 0));

    for (int i = 0; i < (int)wps.size(); ++i)
    {
        colors[i] = image1(inliers_points_src[i].y, inliers_points_src[i].x);
    }

    if (wps.empty())
    {
        cout << "Empty..." << endl;
        cv::waitKey(0);
        return 0;
    }

    cv::viz::WCloud cloud(wps, colors);
    cloud.setRenderingProperty(cv::viz::POINT_SIZE, 4);

    cv::viz::WCoordinateSystem cys(0.2);

    cv::viz::WCameraPosition cp1(K, image1, 0.5);
    cv::viz::WCameraPosition cp2(K, image2, 0.5);

    Matx44d View1a = Matx44d::eye(), View2a = Matx44d::eye();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
        {
            View1a(i, j) = View1(i, j);
            View2a(i, j) = View2(i, j);
        }

    cp1.applyTransform(View1a.inv());
    cp2.applyTransform(View2a.inv());

    cv::viz::Viz3d viewer("Two-View Reconstruction");
    viewer.setWindowSize(Size(1200, 800));
    viewer.setBackgroundColor();
    viewer.showWidget("cys", cys);
    viewer.showWidget("Cloud", cloud);
    viewer.showWidget("cp1", cp1);
    viewer.showWidget("cp2", cp2);


    cout << "Press any key to continue..." << endl;
    cv::waitKey(0);
    viewer.spin();
    exit(0);
}

