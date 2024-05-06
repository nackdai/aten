#pragma once

#include "types.h"
#include "camera/camera.h"

namespace aten {
    class CameraOperator {
    private:
        CameraOperator() = delete;
        ~CameraOperator() = delete;

    public:
        static void move(
            camera& camera,
            int32_t x1, int32_t y1,
            int32_t x2, int32_t y2,
            float scale = float(1));

        static void moveForward(
            camera& camera,
            float offset);
        static void moveRight(
            camera& camera,
            float offset);
        static void moveUp(
            camera& camera,
            float offset);

        static void dolly(
            camera& camera,
            float scale);

        static void rotate(
            camera& camera,
            int32_t width, int32_t height,
            int32_t _x1, int32_t _y1,
            int32_t _x2, int32_t _y2);
    };
}
