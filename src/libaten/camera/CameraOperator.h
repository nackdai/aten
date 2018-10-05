#pragma once

#include "types.h"
#include "camera/camera.h"

namespace aten {
    class CameraOperator {
    private:
        CameraOperator();
        ~CameraOperator();

    public:
        static void move(
            camera& camera,
            int x1, int y1,
            int x2, int y2,
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
            int width, int height,
            int _x1, int _y1,
            int _x2, int _y2);
    };
}