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

        virtual ~AreaLight() = default;

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

        std::shared_ptr<aten::transformable> getLightObject() const
        {
            return m_obj;
        }

        virtual void getSamplePosNormalArea(
            const aten::context& ctxt,
            aten::SamplePosNormalPdfResult* result,
            aten::sampler* sampler) const override final;

    private:
        std::shared_ptr<aten::transformable> m_obj;
    };
}
