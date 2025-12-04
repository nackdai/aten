#pragma once

#include <vector>

#include "light/light.h"
#include "material/diffuse.h"
#include "material/sample_texture.h"
#include "misc/color.h"
#include "renderer/background.h"
#include "scene/host_scene_context.h"

namespace aten {
    class Values;
    class texture;
}

namespace AT_NAME {
    class ImageBasedLight : public Light {
    public:
        ImageBasedLight(const aten::BackgroundResource& bg, const aten::context& ctxt)
            : Light(aten::LightType::IBL, aten::LightAttributeIBL)
        {
            SetBackground(bg, ctxt);
        }

        ImageBasedLight() = delete;
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
            // Sample one point on "hemisphere".
            pdf /= (2.0f * AT_MATH_PI);

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

        static AT_DEVICE_API void sample(
            aten::LightSampleResult& result,
            const aten::LightParameter& param,
            const AT_NAME::context& ctxt,
            const aten::vec3& org,
            const aten::vec3& nml,
            aten::sampler* sampler,
            uint32_t lod)
        {
#if 0
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
#else
            const float r1 = sampler->nextSample();
            const float r2 = sampler->nextSample();
            result.dir = Diffuse::SampleDirection(nml, r1, r2);
#endif

            const auto uv = AT_NAME::Background::ConvertDirectionToUV(result.dir);
            const auto u = uv.x;
            const auto v = uv.y;

            // TODO
            float scene_radius = 10000.0F;
            const auto& scene_bbox = ctxt.GetSceneBoundingBox();
            if (scene_bbox.IsValid()) {
                scene_radius = scene_bbox.ComputeDistanceToCoverBoundingSphere(aten::Deg2Rad(30.0F));
            }

            result.pos = org + scene_radius * result.dir;

            // TODO
            result.nml = -normalize(result.dir);

            // NOTE:
            // Sphere uniform sampling.
            // Sample one point on "hemisphere".
            result.pdf = 1.0f / (2.0f * AT_MATH_PI);

            // NOTE:
            // Theoretically this has to be inf.
            // But, to compute the geometry term with divinding the squared distance without checking the light type,
            // that the distance to light is 1.0 is helpful.
            result.dist_to_light = 1.0F;

            const auto luminance = AT_NAME::sampleTexture(ctxt, param.envmapidx, u, v, aten::vec4(1), lod);

            result.light_color = param.scale * luminance;
        }

    private:
        void preCompute(const std::shared_ptr<aten::texture>& envmap);

        aten::vec3 SampleFromRayWithTexture(
            const aten::ray& in_ray,
            const std::shared_ptr<aten::texture>& envmap) const
        {
            // Translate cartesian coordinates to spherical system.
            auto uv = Background::ConvertDirectionToUV(in_ray.dir);
            return SampleFromUVWithTexture(uv.x, uv.y, envmap);
        }

        aten::vec3 SampleFromUVWithTexture(
            const float u, const float v,
            const std::shared_ptr<aten::texture>& envmap) const
        {
            auto result = envmap->at(u, v);
            return result * bg_.multiplyer;
        }

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
