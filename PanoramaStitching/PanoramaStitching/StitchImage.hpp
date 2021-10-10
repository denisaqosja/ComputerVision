//
//  StitchImage.hpp
//  PanoramaStitching
//
//  Created by Denisa on 4.10.21.
//

#ifndef StitchImage_hpp
#define StitchImage_hpp

#include <stdio.h>
#include "ImageData.h"

class StitchImage
{
public:
    StitchImage();
    void computeHtoref(std::vector<ImageData>& ids, int center);
    cv::Mat3b createStichedImage(std::vector<ImageData>& ids);
    
};

#endif /* StitchImage_hpp */
