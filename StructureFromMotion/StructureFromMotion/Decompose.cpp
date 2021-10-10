//
//  Decompose.cpp
//  StructureFromMotion
//
//  Created by Denisa on 8.10.21.
//

#include "Decompose.hpp"
#include "Triangulate.hpp"

Decompose::Decompose()
{
    
}

void Decompose::decompose(cv::Matx33d E, cv::Matx33d &R1, cv::Matx33d &R2, cv::Vec3d &T1, cv::Vec3d &T2)
{
    //SVD on E
    cv::SVD svdE (E, cv::SVD::FULL_UV);
    cv::Matx33d V = svdE.vt;
    cv::Matx33d U = svdE.u;
    
    //Rotations
    cv::Matx33d W = {0, -1, 0, 1, 0, 0, 0, 0, 1};
    R1 = U * W * V.t();
    R2 = U * W.t() * V.t();
    
    //Translations
    cv::Vec3d S = svdE.u.col(2);
    cv::Vec3d s = S/cv::norm(S, cv::NORM_L2);  //double check the norm, L1 or L2..is it norm?
    T1 = s;
    T2 = -1*s;
    
}

cv::Matx34d Decompose::relativeTransformation(cv::Matx33d E, cv::Matx33d K, std::vector<cv::Point2d> points1, std::vector<cv::Point2d> points2)
{
    cv::Matx33d R1, R2;
    cv::Vec3d T1, T2;
    decompose(E, R1, R2, T1, T2);
    
    // A negative determinant means that R contains a reflection. This is not rigid transformation!
    if (cv::determinant(R1) < 0)
    {
        // scaling the essential matrix by -1 is allowed
        E = -E;
        decompose(E, R1, R2, T1, T2);
    }

    int bestCount = 0;
    
    cv::Matx34d V;
    Triangulate T;

    for (int dR = 0; dR < 2; ++dR)
    {
        cv::Matx33d cR = dR == 0 ? R1 : R2;
        for (int dt = 0; dt < 2; ++dt)
        {
            cv::Matx31d ct = dt == 0 ? T1 : T2;


            cv::Matx34d View1 = cv::Matx34d::eye();
            cv::Matx34d View2;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    View2(i, j) = cR(i, j);
            for (int i = 0; i < 3; ++i)
                View2(i, 3) = ct(i, 0);

            int count = (int)T.triangulate(View1, View2, K, points1, points2).size();
            
            if (count > bestCount)
            {
                V = View2;
                count = bestCount;
            }
        }
        cR = cR.t();
    }

    return V;
}


