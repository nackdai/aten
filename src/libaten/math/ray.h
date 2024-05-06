#pragma once

#include "math/vec3.h"

namespace aten
{
    struct ray
    {
        AT_HOST_DEVICE_API ray()
        {
        }
        AT_HOST_DEVICE_API ray(const vec3& o, const vec3& d)
        {
            dir = normalize(d);
            org = o;
        }
        AT_HOST_DEVICE_API ray(const vec3& o, const vec3& d, const vec3& normal)
        {
            dir = normalize(d);

            AT_ASSERT(dot(dir, normal) >= float(0.0));

            static constexpr float origin = float(1.0) / float(32.0);
            static constexpr float float_scale = float(1.0) / float(65536.0);
            static constexpr float int_scale = float(256.0);

            // A Fast and Robust Method for Avoiding  Self-Intersection
            // Capter 6 in RayTracing Gems

            // 仮数部の上位x[bit]を抽出し、最下位bitにシフトする.
            // 256(=2^8)を掛けるということは指数部に8を足すことになる.
            // その結果の、-127 した結果が抽出する上位x[bit]になる.
            // 例えば、256掛けたあとの指数部が130の場合、x = 130 - 127 => 3[bit].
            // 仮数部には暗黙の1が存在する = > 1.M .
            // それをint32_tにキャストすることで、
            // Mから抽出したx[bit]と暗黙の1を考慮したものが、下位ビットから格納された形で取得できる.
            // ex)
            // float f = 0.015
            //  f : 0x3c75c28f
            //      0011 1100 0111 0101 1100 0010 1000 1111
            //      E = 120, M = 1.111 0101 1100 0010 1000 1111
            // float ff = 256 * f = 256 * 0.015 = 3.83999991 (include 丸め誤差)
            //  ff : 0x4075c28f
            //       0100 0000 0111 0101 1100 0010 1000 1111
            //       E = 128, M = 1.111 0101 1100 0010 1000 1111
            //       e = 128 - 127 = 1
            //       M = [1].[1]11 0101 1100 0010 1000 1111
            //         小数点から上の暗黙の1 + 小数点以下の上位 e (=1) bit
            //         つまり、11 = 3
            // int32_t i = (int32_t)ff = 3
            auto of_ix = (int32_t)(int_scale * normal.x);
            auto of_iy = (int32_t)(int_scale * normal.y);
            auto of_iz = (int32_t)(int_scale * normal.z);

            // NOTE
            // int32_tの場合：負の数は２進数でみたときに値が小さいほどint32_tとしての値が大きくなる（２の補数）.
            // floatの場合：負の数は２進数でみたときでも値が大きいほどfloatとしての値が大きくなる.
            // つまり、int32_tとfloatで負の数の増え方の取り扱いが異なる.
            // 下ではfloatをint32_tとして計算するため、負の数の場合に値が小さくなるような計算をする.
            vec3 p_i(
                intAsFloat(floatAsInt(o.x) + (o.x < float(0) ? -of_ix : of_ix)),
                intAsFloat(floatAsInt(o.y) + (o.y < float(0) ? -of_iy : of_iy)),
                intAsFloat(floatAsInt(o.z) + (o.z < float(0) ? -of_iz : of_iz)));

            org = vec3(
                aten::abs(o.x) < origin ? org.x = o.x + float_scale * normal.x : p_i.x,
                aten::abs(o.y) < origin ? org.y = o.y + float_scale * normal.y : p_i.y,
                aten::abs(o.z) < origin ? org.z = o.z + float_scale * normal.z : p_i.z);
        }

        vec3 org;
        vec3 dir;
    };
}
