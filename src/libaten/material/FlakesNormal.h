#pragma once

#include "material/material.h"

namespace AT_NAME
{
    class FlakesNormal {
    private:
        FlakesNormal() = delete;
        ~FlakesNormal() = delete;

    public:
        static AT_DEVICE_API aten::vec4 gen(
            real u, real v,
            real flake_scale = real(400.0),             // Smaller values zoom into the flake map, larger values zoom out.
            real flake_size = real(0.25),               // Relative size of the flakes
            real flake_size_variance = real(0.7),       // 0.0 makes all flakes the same size, 1.0 assigns random size between 0 and the given flake size
            real flake_normal_orientation = real(0.5)); // Blend between the flake normals (0.0) and the surface normal (1.0)

        static inline AT_DEVICE_API real computeFlakeDensity(
            real flake_size,
            real flakeMapAspect)
        {
            // NOTE
            // size : mapサイズの長辺に占める割合.
            //  ex) w = 1280, h = 720, size = 0.5 -> flake radius = (w > h ? h : w) * size = 720 * 0.5 = 360
            // scale : サイズを 1 / scale にする.

            // NOTE
            // 画面に占めるflakeの面積割合は以下のようになる.
            //     (Pi * radius * radius) * N / (w * h)
            //       radius : flake radius
            //       N : Number of flake
            //       w : map width
            //       h : map height
            // ここで、w > h として、radius = (w > h ? h : w) * size  / scale = size / scale * h を当てはめる.
            //     Density = Pi * (size / scale * h)^2 * N / (w * h)
            //             = Pi * (size / scale)^2 * N * h / w
            // scale = 1 の場合、マップ全体にflakeが約１つとなると考えることができる.
            // つまり、If scale = 1, N = 1 となる.
            // scaleが大きくなっても、これが縦横にscale個繰り返されるだけなので、マップに占めるflakeの面積割合は変わらない.
            //     Density = Pi * (size / scale)^2 * N * h / w
            //             = Pi * size^2 * h / w

            // TODO
            // aspect = w / h の前提なので、h / w を取得したいため逆数にする.
            auto aspect = real(1) / flakeMapAspect;

            auto D = AT_MATH_PI * flake_size * flake_size * aspect;
            D = aten::cmpMin(D, real(1));

            return D;
        }
    };
}
