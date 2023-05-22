#pragma once

#include <vector>

#include "light/light.h"
#include "misc/color.h"
#include "renderer/envmap.h"
#include "material/sample_texture.h"

namespace aten {
    class Values;
}

namespace AT_NAME {
    class ImageBasedLight : public Light {
    public:
        ImageBasedLight()
            : Light(aten::LightType::IBL, aten::LightAttributeIBL)
        {}
        ImageBasedLight(const std::shared_ptr<AT_NAME::envmap>& envmap)
            : Light(aten::LightType::IBL, aten::LightAttributeIBL)
        {
            setEnvMap(envmap);
        }

        ImageBasedLight(aten::Values& val);

        virtual ~ImageBasedLight() = default;

    public:
        void setEnvMap(const std::shared_ptr<AT_NAME::envmap>& envmap)
        {
            if (m_envmap != envmap) {
                m_envmap = envmap;
                m_param.envmapidx = envmap->getTexture()->id();

                preCompute();
            }
        }

        std::shared_ptr<const AT_NAME::envmap> getEnvMap() const
        {
            return m_envmap;
        }

        real samplePdf(const aten::ray& r) const;

        static AT_DEVICE_API real samplePdf(const aten::vec3& clr, real avgIllum)
        {
            auto illum = AT_NAME::color::luminance(clr);

            auto pdf = illum / avgIllum;

            // NOTE
            // 半径１の球の面積で割る.
            pdf /= (4 * AT_MATH_PI);

            return pdf;
        }

        real getAvgIlluminace() const
        {
            return m_avgIllum;
        }

        aten::LightSampleResult sample(
            const aten::context& ctxt,
            const aten::vec3& org,
            const aten::vec3& nml,
            aten::sampler* sampler) const;

        template <typename CONTEXT>
        static AT_DEVICE_MTRL_API void sample(
            aten::LightSampleResult& result,
            const aten::LightParameter& param,
            const CONTEXT& ctxt,
            const aten::vec3& org,
            const aten::vec3& nml,
            aten::sampler* sampler,
            uint32_t lod)
        {
            const auto& n = nml;
            const auto t = aten::getOrthoVector(n);
            const auto b = normalize(cross(n, t));

            const real r1 = sampler->nextSample();
            const real r2 = sampler->nextSample();

            const real sinpsi = aten::sin(2 * AT_MATH_PI * r1);
            const real cospsi = aten::cos(2 * AT_MATH_PI * r1);
            const real costheta = aten::pow(1 - r2, 0.5);
            const real sintheta = aten::sqrt(1 - costheta * costheta);

            result.dir = normalize(t * sintheta * cospsi + b * sintheta * sinpsi + n * costheta);

            const auto uv = AT_NAME::envmap::convertDirectionToUV(result.dir);
            const auto u = uv.x;
            const auto v = uv.y;

            // TODO
            // シーンのAABBを覆う球上に配置されるようにするべき.
            result.pos = org + real(100000) * result.dir;
            result.pdf = dot(nml, result.dir) / AT_MATH_PI;

#ifdef __CUDACC__
            // envmapidx is index to array of textures in context.
            // In GPU, sampleTexture requires texture id of CUDA. So, arguments is different.
            const auto le = tex2DLod<float4>(ctxt.textures[param.envmapidx], u, v, lod);
#else
            const auto le = AT_NAME::sampleTexture(param.envmapidx, u, v, aten::vec4(1), lod);
#endif

            result.le = aten::vec3(le.x, le.y, le.z);
            result.finalColor = result.le;
        }

    private:
        void preCompute();

    private:
        std::shared_ptr<envmap> m_envmap;

        real m_avgIllum{ real(0) };

        // v方向のcdf(cumulative distribution function = 累積分布関数 = sum of pdf).
        std::vector<real> m_cdfV;

        // u方向のcdf(cumulative distribution function = 累積分布関数 = sum of pdf).
        // 列ごとに保持する.
        std::vector<std::vector<real>> m_cdfU;
    };
}
