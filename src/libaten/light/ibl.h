#pragma once

#include <vector>

#include "light/light.h"
#include "misc/color.h"
#include "renderer/envmap.h"

namespace aten {
    class Values;
}

namespace AT_NAME {
    class ImageBasedLight : public Light {
    public:
        ImageBasedLight()
            : Light(aten::LightType::IBL, LightAttributeIBL)
        {}
        ImageBasedLight(const std::shared_ptr<AT_NAME::envmap>& envmap)
            : Light(aten::LightType::IBL, LightAttributeIBL)
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

        virtual aten::LightSampleResult sample(
            const aten::context& ctxt,
            const aten::vec3& org,
            aten::sampler* sampler) const override final
        {
            AT_ASSERT(false);
            return aten::LightSampleResult();
        }

        virtual aten::LightSampleResult sample(
            const aten::context& ctxt,
            const aten::vec3& org,
            const aten::vec3& nml,
            aten::sampler* sampler) const override final;

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
