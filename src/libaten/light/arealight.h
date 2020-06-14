#pragma once

#include "light/light.h"
#include "geometry/transformable.h"

namespace AT_NAME {
    class AreaLight : public Light {
    public:
        AreaLight()
            : Light(aten::LightType::Area, LightAttributeArea)
        {}
        AreaLight(aten::transformable* obj, const aten::vec3& le)
            : Light(aten::LightType::Area, LightAttributeArea)
        {
            m_obj = obj;

            m_param.objid = obj->id();
            m_param.le = le;
        }

        AreaLight(aten::Values& val)
            : Light(aten::LightType::Area, LightAttributeArea, val)
        {
            m_obj = (aten::transformable*)val.get("object", m_obj);

            m_param.objid = (m_obj ? m_obj->id() : -1);
            m_param.le = val.get("color", m_param.le);
        }

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
            result->intensity = 1;
            result->finalColor = param->le;
        }

        virtual aten::LightSampleResult sample(
            const aten::context& ctxt,
            const aten::vec3& org,
            aten::sampler* sampler) const override final;

        virtual const aten::hitable* getLightObject() const override final
        {
            return (aten::hitable*)m_obj;
        }

        virtual void getSamplePosNormalArea(
            const aten::context& ctxt,
            aten::hitable::SamplePosNormalPdfResult* result,
            aten::sampler* sampler) const override final
        {
            if (m_obj) {
                auto obj = getLightObject();
                return obj->getSamplePosNormalArea(ctxt, result, sampler);
            }
        }

    private:
        aten::transformable* m_obj{ nullptr };
    };
}
