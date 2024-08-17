#pragma once

#include "types.h"
#include "camera/camera.h"

namespace aten {
    class CameraOperator {
    public:
        CameraOperator() = delete;
        ~CameraOperator() = delete;

        CameraOperator(const CameraOperator&) = delete;
        CameraOperator& operator=(const CameraOperator&) = delete;
        CameraOperator(CameraOperator&&) = delete;
        CameraOperator& operator=(CameraOperator&&) = delete;

        static void move(
            Camera& camera,
            int32_t x1, int32_t y1,
            int32_t x2, int32_t y2,
            float scale = float(1));

        static void MoveForward(
            Camera& camera,
            float offset);
        static void MoveRight(
            Camera& camera,
            float offset);
        static void MoveUp(
            Camera& camera,
            float offset);

        static void Dolly(
            Camera& camera,
            float scale);

        static void Rotate(
            Camera& camera,
            int32_t width, int32_t height,
            int32_t _x1, int32_t _y1,
            int32_t _x2, int32_t _y2);
    };
}
