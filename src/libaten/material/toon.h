#pragma once

#include <functional>
#include "material/material.h"
#include "light/light.h"

namespace aten {
    class Values;
}

namespace AT_NAME
{
    class toon : public NPRMaterial {
    public:
        using ComputeToonShadeFunc = std::function<real(real)>;

        toon(const aten::vec3& e, std::shared_ptr<AT_NAME::Light> light)
            : NPRMaterial(aten::MaterialType::Toon, e, light)
        {
        }
        toon(const aten::vec3& e, std::shared_ptr<AT_NAME::Light> light, ComputeToonShadeFunc func)
            : NPRMaterial(aten::MaterialType::Toon, e, light)
        {
            setComputeToonShadeFunc(func);
        }

        toon(aten::Values& val);

        virtual ~toon() {}

    public:
        virtual real pdf(
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v) const override final
        {
            return real(1);
        }

        virtual aten::vec3 sampleDirection(
            const aten::ray& ray,
            const aten::vec3& normal,
            real u, real v,
            aten::sampler* sampler) const override final;

        virtual aten::vec3 bsdf(
            const aten::vec3& normal,
            const aten::vec3& wi,
            const aten::vec3& wo,
            real u, real v) const override final;

        virtual MaterialSampling sample(
            const aten::ray& ray,
            const aten::vec3& normal,
            const aten::vec3& orgnormal,
            aten::sampler* sampler,
            real u, real v,
            bool isLightPath = false) const override final;

        void setComputeToonShadeFunc(ComputeToonShadeFunc func)
        {
            m_func = func;
        }

    private:
        virtual aten::vec3 bsdf(
            real cosShadow,
            real u, real v) const override final;

    private:
        // TODO
        // GPGPU化する場合は考えないといけない....
        ComputeToonShadeFunc m_func;
    };
}
