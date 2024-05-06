#include "camera/CameraOperator.h"
#include "math/vec4.h"
#include "math/mat4.h"

namespace aten {
    void CameraOperator::move(
        camera& camera,
        int32_t x1, int32_t y1,
        int32_t x2, int32_t y2,
        float scale/*= float(1)*/)
    {
        auto& pos = camera.getPos();
        auto& at = camera.getAt();

        float offsetX = (float)(x1 - x2);
        offsetX *= scale;

        float offsetY = (float)(y1 - y2);
        offsetY *= scale;

        // 移動ベクトル.
        aten::vec3 offset(offsetX, offsetY, float(0));

        // カメラの回転を考慮する.
        aten::vec3 dir = at - pos;
        dir = normalize(dir);
        dir.y = float(0);

        aten::mat4 mtxRot;
        mtxRot.asRotateFromVector(dir, aten::vec3(0, 1, 0));

        mtxRot.applyXYZ(offset);

        pos += offset;
        at += offset;
    }

    void CameraOperator::moveForward(
        camera& camera,
        float offset)
    {
        // カメラの向いている方向(Z軸)に沿って移動.

        auto& pos = camera.getPos();
        auto& at = camera.getAt();

        auto dir = camera.getDir();
        dir *= offset;

        pos += dir;
        at += dir;
    }

    void CameraOperator::moveRight(
        camera& camera,
        float offset)
    {
        // カメラの向いている方向の右軸(-X軸(右手座標なので))に沿って移動.
        auto vz = camera.getDir();

        vec3 vup(0, 1, 0);
        if (aten::abs(vz.x) < AT_MATH_EPSILON && aten::abs(vz.z) < AT_MATH_EPSILON) {
            // UPベクトルとの外積を計算できないので、
            // 新しいUPベクトルをでっちあげる・・・
            vup = vec3(float(0), float(0), -vz.y);
        }

        auto vx = cross(vup, vz);
        vx = normalize(vx);

        vx *= offset;
        vx *= float(-1);    // -X軸に変換.

        auto& pos = camera.getPos();
        auto& at = camera.getAt();

        pos += vx;
        at += vx;
    }

    void CameraOperator::moveUp(
        camera& camera,
        float offset)
    {
        // カメラの向いている方向の右軸(Y軸)に沿って移動.
        auto vz = camera.getDir();

        vec3 vup(0, 1, 0);
        if (aten::abs(vz.x) < AT_MATH_EPSILON && aten::abs(vz.z) < AT_MATH_EPSILON) {
            // UPベクトルとの外積を計算できないので、
            // 新しいUPベクトルをでっちあげる・・・
            vup = vec3(float(0), float(0), -vz.y);
        }

        auto vx = cross(vup, vz);
        vx = normalize(vx);

        auto vy = cross(vz, vx);

        vy *= offset;

        auto& pos = camera.getPos();
        auto& at = camera.getAt();

        pos += vy;
        at += vy;
    }

    void CameraOperator::dolly(
        camera& camera,
        float scale)
    {
        auto& pos = camera.getPos();
        auto& at = camera.getAt();

        // 視点と注視点の距離.
        float len = length(pos - at);

        // 視点から注視点への方向.
        auto dir = pos - at;
        dir = normalize(dir);

        // スケーリング.
        // 注視点までの距離に応じる.
        auto distScale = scale * len * 0.01f;
        dir *= distScale;

        // 新しい視点.
        pos += dir;
    }

    static inline float projectionToSphere(
        float radius,
        float x,
        float y)
    {
        float z = float(0);
        float dist = aten::sqrt(x * x + y * y);

        // r * 1/√2 の点で双曲線と接する内側と外側

        if (dist < radius * float(0.70710678118654752440)) {
            // 内側

            // NOTE
            // r * r = x * x + y * y + z * z
            // <=> z * z = r * r - (x * x + y * y)
            z = aten::sqrt(radius * radius - dist * dist);
        }
        else {
            // 外側
            float t = radius * radius * 0.5f;
            z = t / dist;
        }

        return z;
    }

    static inline float normalizeHorizontal(int32_t x, float width)
    {
        float ret = (float(2) * x - width) / width;
        return ret;
    }

    static inline float normalizeVertical(int32_t y, float height)
    {
        float ret = (height - float(2) * y) / height;
        return ret;
    }

    void CameraOperator::rotate(
        camera& camera,
        int32_t width, int32_t height,
        int32_t _x1, int32_t _y1,
        int32_t _x2, int32_t _y2)
    {
        static const float radius = float(0.8);

        float x1 = normalizeHorizontal(_x1, (float)width);
        float y1 = normalizeVertical(_y1, (float)height);

        float x2 = normalizeHorizontal(_x2, (float)width);
        float y2 = normalizeVertical(_y2, (float)height);

        // スクリーン上の２点からトラックボール上の点を計算する.
        // GLUTと同じ方法.

        aten::vec3 v1(
            x1, y1,
            projectionToSphere(radius, x1, y1));
        v1 = normalize(v1);

        aten::vec3 v2(
            x2, y2,
            projectionToSphere(radius, x2, y2));
        v2 = normalize(v2);

        // 回転軸.
        auto axis = cross(v1, v2);
        axis = normalize(axis);

        const auto dir = camera.getDir();
        aten::mat4 transform;
        transform.asRotateFromVector(dir, aten::vec3(0, 1, 0));

        // カメラの回転状態に合わせて軸も回転.
        transform.applyXYZ(axis);

        // 回転の角度
        // NOTE
        // V1・V2 = |V1||V2|cosθ = cosθ (|V1| = |V2| = 1)
        // θ = acos(cosθ)
        // => θ = acos(cosθ) = acos(V1・V2)
        float theta = aten::acos(dot(v1, v2));

        // 回転.
        aten::mat4 mtxRot;
        mtxRot.asRotateByAxis(theta, axis);

        auto& pos = camera.getPos();
        auto& at = camera.getAt();

        pos -= at;
        pos = mtxRot.applyXYZ(pos);
        pos += at;
    }
}
