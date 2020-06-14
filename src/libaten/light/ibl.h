#pragma once

#include <vector>
#include "light/light.h"
#include "renderer/envmap.h"
#include "misc/color.h"

namespace AT_NAME {
    class ImageBasedLight : public Light {
    public:
        ImageBasedLight()
            : Light(aten::LightType::IBL, LightAttributeIBL)
        {}
        ImageBasedLight(AT_NAME::envmap* envmap)
            : Light(aten::LightType::IBL, LightAttributeIBL)
        {
            setEnvMap(envmap);
        }

        ImageBasedLight(aten::Values& val)
            : Light(aten::LightType::IBL, LightAttributeIBL, val)
        {
            aten::texture* tex = (aten::texture*)val.get("envmap", nullptr);

            AT_NAME::envmap* bg = new AT_NAME::envmap();
            bg->init(tex);

            setEnvMap(bg);
        }

        virtual ~ImageBasedLight() {}

    public:
        void setEnvMap(AT_NAME::envmap* envmap)
        {
            if (m_param.envmap.ptr != envmap) {
                m_param.envmap.ptr = envmap;

                preCompute();
            }
        }

        const AT_NAME::envmap* getEnvMap() const
        {
            return (AT_NAME::envmap*)m_param.envmap.ptr;
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
            return std::move(aten::LightSampleResult());
        }

        virtual aten::LightSampleResult sample(
            const aten::context& ctxt,
            const aten::vec3& org,
            const aten::vec3& nml,
            aten::sampler* sampler) const override final;

    private:
        void preCompute();

    private:
        real m_avgIllum{ real(0) };

        // v方向のcdf(cumulative distribution function = 累積分布関数 = sum of pdf).
        std::vector<real> m_cdfV;

        // u方向のcdf(cumulative distribution function = 累積分布関数 = sum of pdf).
        // 列ごとに保持する.
        std::vector<std::vector<real>> m_cdfU;
    };
}
