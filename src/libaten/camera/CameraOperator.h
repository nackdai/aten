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
            real scale = real(1));

        static void moveForward(
            camera& camera,
            real offset);
        static void moveRight(
            camera& camera,
            real offset);
        static void moveUp(
            camera& camera,
            real offset);

        static void dolly(
            camera& camera,
            real scale);

        static void rotate(
            camera& camera,
            int32_t width, int32_t height,
            int32_t _x1, int32_t _y1,
            int32_t _x2, int32_t _y2);
    };
}
