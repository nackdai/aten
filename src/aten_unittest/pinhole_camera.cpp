#include <gtest/gtest.h>

#include "aten.h"
#include "camera/pinhole.h"

TEST(pinhole_camera_test, ComputePixelWidthAtDistanceTest) {
    aten::CameraParameter param;
    param.vfov = real(60);
    param.width = 1280;
    param.height = 720;

    auto result = aten::PinholeCamera::computePixelWidthAtDistance(param, real(1));

    EXPECT_FLOAT_EQ(result, real(0.000473979220));
}
