#include "renderer/npr/feature_line.h"
#include "geometry/EvaluateHitResult.h"

namespace aten {
namespace npr {
    aten::vec3 FeatureLine::RenderFeatureLine(
        const aten::vec3& color,
        int32_t x, int32_t y,
        int32_t width, int32_t height,
        const aten::hitrecord& hrec,
        const context& ctxt,
        const scene& scene,
        const Camera& camera)
    {
        // NOTE
        // Ray Tracing NPR-Style Feature Lines
        // http://www.sci.utah.edu/publications/choudhury09/NPR-lines.NPAR09.pdf

        constexpr auto M = 24;
        constexpr auto N = 2;
        constexpr auto radius = float(2.0);

        uint32_t edge_strengh_measurement = 0;

        for (int32_t n = 0; n < N; n++) {
            const auto sample_in_n = (N - n) * 8;
            for (int32_t i = 0; i < sample_in_n; i++) {
                const auto theta = (AT_MATH_PI_2 * i) / sample_in_n;
                const auto stencil_pos_x = radius * (N - n) * aten::cos(theta);
                const auto stencil_pos_y = radius * (N - n) * aten::sin(theta);

                const auto u = (x + stencil_pos_x) / float(width);
                const auto v = (y + stencil_pos_y) / float(height);

                // TODO
                // Specify sampler as nullptr forcibly at this moment...
                const auto stencil_ray = camera.sample(u, v, nullptr).r;

                aten::Intersection isect;

                if (scene.hit(ctxt, stencil_ray, AT_MATH_EPSILON, AT_MATH_INF, isect)) {
                    aten::hitrecord hrec_stencil;
                    const auto& obj = ctxt.GetObject(isect.objid);
                    AT_NAME::evaluate_hit_result(hrec_stencil, obj, ctxt, stencil_ray, isect);

                    if (hrec.meshid != hrec_stencil.meshid) {
                        edge_strengh_measurement++;
                    }
                    else {
                        const auto d = dot(hrec.normal, hrec_stencil.normal);
                        if (d < float(0.25)) {
                            edge_strengh_measurement++;
                        }
                    }
                }
            }
        }

        auto edge_strengh = aten::clamp(
            1.0f - aten::abs(edge_strengh_measurement - M * float(0.5)) / (M * float(0.5)),
            0.0f, 1.0f);
        const auto result = color * (float(1) - edge_strengh);
        return result;
    }
}
}
