#include <gtest/gtest.h>

#include "aten.h"
#include "camera/pinhole.h"

TEST(pinhole_camera_test, ComputePixelWidthAtDistanceTest)
{
    aten::CameraParameter param;
    param.vfov = 60;
    param.width = 1280;
    param.height = 720;

    auto result = aten::PinholeCamera::computePixelWidthAtDistance(param, 1.0f);

    EXPECT_FLOAT_EQ(result, 0.000473979220);
}
