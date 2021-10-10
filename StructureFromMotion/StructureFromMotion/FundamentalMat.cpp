//
//  FundamentalMat.cpp
//  StructureFromMotion
//
//  Created by Denisa on 5.10.21.
//

#include "FundamentalMat.hpp"

FundamentalMat::FundamentalMat(int iter, int t)
{
    iterations = iter;
    threshold = t;
}

cv::Matx33d FundamentalMat::computeF(std::vector<cv::Point2d> points_src, std::vector<cv::Point2d> points_trg)
{
    // 8-point Algorithm
    
    //1. Construct A(8x9)
    cv::Mat1d A(8,9);

    for(int i = 0; i < 8; i++)
    {
        A[i][0] = points_src[i].x * points_trg[i].x;
        A[i][1] = points_src[i].x * points_trg[i].y;
        A[i][2] = points_src[i].x;
        A[i][3] = points_src[i].y * points_trg[i].x;
        A[i][4] = points_src[i].y * points_trg[i].y;
        A[i][5] = points_src[i].y;
        A[i][6] = points_trg[i].x;
        A[i][7] = points_trg[i].y;
        A[i][8] = 1;
    }
    
    //2. SVD on A, to solve Af=0
    cv::SVD svdA(A, cv::SVD::FULL_UV);  
    cv::Mat1d V = svdA.vt.t();
    cv::Mat1d f = V.col(8);
    
    //3. Construct F
    cv::Matx33d F;
    
    for(int x = 0; x < 3; x++)
        for (int y = 0; y < 3; y++)
            F(x,y) = f(3 * x + y);
    
    // Compute SVD on F
    cv::SVD svdF(F, cv::SVD::FULL_UV);
    cv::Matx33d S = cv::Mat::diag(svdF.w);
    //Ensure: rank(F) = 2, set S(3x3)
    S(2,2) = 0;
    
    //4.Recompute F and normalize it
    F = (cv::Matx33d(svdF.u) * S * cv::Matx33d(svdF.vt)).t();
    F *= (1.0/F(2,2));
    
    return F;
}


std::vector<int> FundamentalMat::numInliners(cv::Matx33d F, std::vector<cv::Point2d> points_src, std::vector<cv::Point2d> points_trg)
{
    int inliers = 0;
    std::vector<int> inliers_ids;
    
    for (int i = 0; i < points_src.size(); i++)
    {
        cv::Vec3d x1_homogen(points_src[i].x, points_src[i].y, 1);
        cv::Vec3d x2_homogen(points_trg[i].x, points_trg[i].y, 1);
        
        cv::Vec3d epipolar_line = F * x1_homogen;
        cv::Vec3d epipolar_line_norm = epipolar_line / (powf(epipolar_line(0),2) + powf(epipolar_line(1),2));
        
        //points_trg should lie in the epipolar line
        double distance;
        distance = x2_homogen.dot(epipolar_line_norm);
        
        if(distance < threshold)
        {
            inliers++;
            inliers_ids.push_back(i);
        }
    }
    return inliers_ids;
}

cv::Matx33d FundamentalMat::FRansac(std::vector<cv::Point2d> points_src, std::vector<cv::Point2d> points_trg, std::vector<int> &inliersId)
{
    //randomly generate 8 points
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution{0, (int)points_src.size()-1};
    
    int num_points = 8, bestInliers = 0;
    cv::Matx33d F, bestF;
    std::vector<int> best_inliersId, inliers_id;
    
    for(int iter = 0; iter < iterations; iter++)
    {
        std::vector<cv::Point2d> subset_src, subset_trg;
        inliers_id.empty();
        
        for(int point = 0; point<num_points; point++)
        {
            int random_idx = distribution(gen);
            subset_src.push_back(points_src[random_idx]);
            subset_trg.push_back(points_trg[random_idx]);
        }
        
        //Calculate Fundamental matrix
        F = computeF(subset_src, subset_trg);
        
        //epipolar constraint: will give the num of inliers
        inliers_id = numInliners(F, points_src, points_trg);
        
        if((int)inliers_id.size() > bestInliers)
        {
            bestInliers = (int)inliers_id.size();
            bestF = F;
            best_inliersId = inliers_id;
        }
    }
    inliersId = best_inliersId;
    
    return bestF;
}

void FundamentalMat::testFundamentalMat()
{
    std::vector<cv::Point2d> points1 = {{1, 1}, {3, 7}, {2, -5}, {10, 11}, {11, 2}, {-3, 14}, {236, -514}, {-5, 1}};
    std::vector<cv::Point2d> points2 = {{25, 156},   {51, -83}, {-144, 5},  {345, 15},
                                    {215, -156}, {151, 83}, {1544, 15}, {451, -55}};
    auto F = computeF(points1, points2);
    std::cout << "Testing Fundamental Matrix..." << std::endl << "Your result:" << std::endl;
    std::cout << F << std::endl;

    cv::Matx33d Href = {0.001260822171230067,  0.0001614643951166923, -0.001447955678643285,
                 -0.002080014358205309, -0.002981504896782918, 0.004626528742122177,
                 -0.8665185546662642,   -0.1168790312603214,   1};
    std::cout << "Reference: " << std::endl << Href << std::endl;

    auto error = Href - F;
    double e   = norm(error);
    std::cout << "Error: " << e << std::endl;
    if (e < 1e-10)
        std::cout << "Test: SUCCESS!" << std::endl;
    else
        std::cout << "Test: FAIL!" << std::endl;
    std::cout << "============================" << std::endl;
}
