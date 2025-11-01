#include "math/mat4.h"

namespace aten {
    const mat4 mat4::Identity(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1);

    const mat4 mat4::Zero(
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0);

    aten::tuple<aten::vec3, aten::mat4, aten::vec3> mat4::Decompose() const
    {
        // Gram-Schmidt orthogonalization
        aten::vec3 c0{ v[0] };
        aten::vec3 c1{ v[1] };
        aten::vec3 c2{ v[2] };

        aten::vec3 out_scale;
    
        out_scale.x = length(c0);
        auto r0 = c0 / out_scale.x;

        c1 -= r0 * dot(r0, c1);
        out_scale.y = length(c1);
        auto r1 = c1 / out_scale.y;

        c2 -= r0 * dot(r0, c2);
        c2 -= r1 * dot(r1, c2);
        out_scale.z = glm::length(c2);
        auto r2 = c2 / out_scale.z;

        auto out_rot = aten::mat4(r0, r1, r2);

        float det = out_rot.DeterminantAs3x3();
        if (det < 0) {
            out_scale *= -1.0f;
            out_rot = -out_rot;
        }

        aten::vec3 out_trans{
            v[0].w,
            v[1].w,
            v[2].w,
        };

        return aten::make_tuple(out_scale, out_rot, out_trans);
    }

    float mat4::DeterminantAs3x3() const
    {
        return
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
            m[1][0] * (m[0][1] * m[2][2] - m[0][2] * m[2][1]) +
            m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1]);
    }
}
