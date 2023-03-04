#pragma once

#include <memory>

#include "geometry/transformable.h"
#include "light/light.h"

namespace aten {
    class Values;
}

namespace AT_NAME {
    class AreaLight : public Light {
    public:
        AreaLight()
            : Light(aten::LightType::Area, LightAttributeArea)
        {}
        AreaLight(const std::shared_ptr<aten::transformable>& obj, const aten::vec3& le)
            : Light(aten::LightType::Area, LightAttributeArea)
        {
            m_obj = obj;

            m_param.objid = obj->id();
            m_param.le = le;
        }

        AreaLight(const aten::Values& val);

        virtual ~AreaLight() {}

    public:
        static AT_DEVICE_API void sample(
            const aten::hitrecord* rec,
            const aten::LightParameter* param,
            const aten::vec3& org,
            aten::sampler* sampler,
            aten::LightSampleResult* result)
        {
            result->pos = rec->p;

            // TODO
            // AMDのProRender(Baikal)ではこ dist2/面積 となっているが...
            auto dist2 = aten::squared_length(rec->p - org);
            result->pdf = 1 / rec->area;

            result->dir = rec->p - org;
            result->nml = rec->normal;

            result->le = param->le;
            result->finalColor = param->le;
        }

        virtual aten::LightSampleResult sample(
            const aten::context& ctxt,
            const aten::vec3& org,
            aten::sampler* sampler) const override final;

        virtual std::shared_ptr<const aten::hitable> getLightObject() const override final
        {
            return std::static_pointer_cast<aten::hitable>(m_obj);
        }

        virtual void getSamplePosNormalArea(
            const aten::context& ctxt,
            aten::SamplePosNormalPdfResult* result,
            aten::sampler* sampler) const override final
        {
            if (m_obj) {
                auto obj = getLightObject();
                return obj->getSamplePosNormalArea(ctxt, result, sampler);
            }
        }

    private:
        std::shared_ptr<aten::transformable> m_obj;
    };
}
