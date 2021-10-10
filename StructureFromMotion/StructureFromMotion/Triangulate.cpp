//
//  Triangulate.cpp
//  StructureFromMotion
//
//  Created by Denisa on 5.10.21.
//

#include "Triangulate.hpp"

using namespace std;
using namespace cv;

Triangulate::Triangulate(){}

cv::Point3f Triangulate::triangulatePoints(cv::Matx34d P1, cv::Matx34d P2, cv::Point2d p1, cv::Point2d p2)
{
    //matrix A
    cv::Matx44d A;
    A={p1.x*P1(2,0)-P1(0,0), p1.x*P1(2,1)-P1(0,1), p1.x*P1(2,2)-P1(0,2), p1.x*P1(2,3)-P1(0,3),
        p1.y*P1(2,0)-P1(1,0), p1.y*P1(2,1)-P1(1,1), p1.y*P1(2,2)-P1(1,2), p1.y*P1(2,3)-P1(1,3),
        p2.x*P2(2,0)-P2(0,0), p2.x*P2(2,1)-P2(0,1), p2.x*P2(2,2)-P2(0,2), p2.x*P2(2,3)-P2(0,3),
        p2.y*P2(2,0)-P2(1,0), p2.y*P2(2,1)-P2(1,1), p2.y*P2(2,2)-P2(1,2), p2.y*P2(2,3)-P2(1,3)
    };
    
    //SVD on A
    cv::SVD svdA(A, cv::SVD::FULL_UV);
    cv::Mat1d V = svdA.vt.t();
    cv::Mat1d X = V.col(3);
    cv::Point3f x = {(float)(X(0)/X(3)), (float)(X(1)/X(3)), (float)(X(2)/X(3))};
    
    return x;
}

std::vector<cv::Point3f> Triangulate::triangulate(cv::Matx34d View1, cv::Matx34d View2, cv::Matx33d K, vector<cv::Point2d> points1, vector<cv::Point2d> points2)
{
    std::vector<cv::Point3f> wps;

    cv::Matx34d P1, P2;
    P1 = K * View1;
    P2 = K * View2;
    
    for (int i = 0; i < (int)points1.size(); ++i)
    {
        cv::Point3f wp = triangulatePoints(P1, P2, points1[i], points2[i]);

        // check if this point is in front of both cameras
        cv::Vec4d ptest(wp.x, wp.y, wp.z, 1);
        cv::Vec3d p1 = K * View1 * ptest;
        cv::Vec3d p2 = K * View2 * ptest;

        if (p1[2] > 0 && p2[2] > 0)
        {
            wps.push_back(wp);
        }
    }

    return wps;
    
}


void Triangulate::testTriangulate()
{
    cv::Matx34d P1 = {1, 2, 3, 6, 4, 5, 6, 37, 7, 8, 9, 15};
    cv::Matx34d P2 = {51, 12, 53, 73, 74, 15, -6, -166, 714, -8, 95, 16};

    auto F = triangulatePoints(P1, P2, cv::Point2f(14, 267), cv::Point2f(626, 67));
    cout << "Testing Triangulation..." << endl << "Your result:" << endl;
    cout << F << endl;

    cv::Point3f wpref = {0.782409, 3.89115, -5.72358};
    cout << "Reference: " << endl << wpref << endl;

    auto error = wpref - F;
    double e   = norm(error);
    cout << "Error: " << e << endl;
    if (e < 1e-5)
        cout << "Test: SUCCESS!" << endl;
    else
        cout << "Test: FAIL!" << endl;
    cout << "============================" << endl;
}

