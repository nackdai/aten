#pragma once

#include <vector>

#include "light/light.h"
#include "misc/color.h"
#include "material/sample_texture.h"
#include "renderer/background.h"

namespace aten {
    class Values;
    class texture;
}

namespace AT_NAME {
    class ImageBasedLight : public Light {
    public:
        ImageBasedLight()
            : Light(aten::LightType::IBL, aten::LightAttributeIBL)
        {}
        ImageBasedLight(const aten::BackgroundResource& bg, const aten::context& ctxt)
            : Light(aten::LightType::IBL, aten::LightAttributeIBL)
        {
            SetBackground(bg, ctxt);
        }

        ImageBasedLight(aten::Values& val);

        virtual ~ImageBasedLight() = default;

    public:
        void SetBackground(
            const aten::BackgroundResource& bg,
            const aten::context& ctxt)
        {
            if (bg_.envmap_tex_idx != bg.envmap_tex_idx) {
                bg_ = bg;
                m_param.envmapidx = bg.envmap_tex_idx;

                AT_ASSERT(bg.envmap_tex_idx >= 0);
                auto envmap = ctxt.GetTexture(bg.envmap_tex_idx);
                preCompute(envmap);
            }
        }

        float samplePdf(const aten::ray& r, const aten::context& ctxt) const;

        static AT_HOST_DEVICE_API float samplePdf(const aten::vec3& clr, float avgIllum)
        {
            auto illum = AT_NAME::color::luminance(clr);

            auto pdf = illum / avgIllum;

            // NOTE:
            // Sphere uniform sampling.
            // Sample one point on sphere.
            pdf /= (4.0f * AT_MATH_PI);

            return pdf;
        }

        float getAvgIlluminace() const
        {
            return m_avgIllum;
        }

        aten::LightSampleResult sample(
            const aten::context& ctxt,
            const aten::vec3& org,
            const aten::vec3& nml,
            aten::sampler* sampler) const;

        template <class CONTEXT>
        static AT_DEVICE_API void sample(
            aten::LightSampleResult& result,
            const aten::LightParameter& param,
            const CONTEXT& ctxt,
            const aten::vec3& org,
            const aten::vec3& nml,
            aten::sampler* sampler,
            uint32_t lod)
        {
            const auto& n = nml;

            aten::vec3 t, b;
            aten::tie(t, b) = aten::GetTangentCoordinate(n);

            const float r1 = sampler->nextSample();
            const float r2 = sampler->nextSample();

            const auto costheta = aten::sqrt(1 - r1);
            const auto sintheta = aten::sqrt(r1);

            const auto phi = AT_MATH_PI_2 * r2;
            const auto cosphi = aten::cos(phi);
            const auto sinphi = aten::sin(phi);

            result.dir = normalize(t * sintheta * cosphi + b * sintheta * sinphi + n * costheta);

            const auto uv = AT_NAME::Background::ConvertDirectionToUV(result.dir);
            const auto u = uv.x;
            const auto v = uv.y;

            // TODO
            // Sphere size should cover entire scene.
            result.pos = org + float(100000) * result.dir;

            // NOTE:
            // Sphere uniform sampling.
            // Sample one point on sphere.
            result.pdf = 1.0f / (4.0f * AT_MATH_PI);

            // envmapidx is index to array of textures in context.
            // In GPU, sampleTexture requires texture id of CUDA. So, arguments is different.
            auto envmapidx =
#ifdef __CUDACC__
                ctxt.textures[param.envmapidx];
#else
                param.envmapidx;
#endif

            const auto luminance = AT_NAME::sampleTexture(envmapidx, u, v, aten::vec4(1), lod);

            result.light_color = param.scale * luminance;
        }

    private:
        void preCompute(const std::shared_ptr<aten::texture>& envmap);

    private:
        aten::BackgroundResource bg_;

        float m_avgIllum{ float(0) };

        // v方向のcdf(cumulative distribution function = 累積分布関数 = sum of pdf).
        std::vector<float> m_cdfV;

        // u方向のcdf(cumulative distribution function = 累積分布関数 = sum of pdf).
        // 列ごとに保持する.
        std::vector<std::vector<float>> m_cdfU;
    };
}
