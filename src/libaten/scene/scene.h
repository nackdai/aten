#pragma once

#include <vector>
#include <algorithm>
#include <iterator>

#include "accelerator/accelerator.h"
#include "accelerator/bvh.h"
#include "light/light.h"
#include "light/ibl.h"
#include "scene/context.h"

namespace AT_NAME {
    class scene {
    public:
        scene() {}
        virtual ~scene() {}

    public:
        virtual void build(const aten::context& ctxt)
        {}

        void add(aten::hitable* s)
        {
            m_list.push_back(s);
        }

        virtual bool hit(
            const aten::context& ctxt,
            const aten::ray& r,
            real t_min, real t_max,
            bool enableLod,
            aten::hitrecord& rec,
            aten::Intersection& isect) const = 0;

        bool hit(
            const aten::context& ctxt,
            const aten::ray& r,
            real t_min, real t_max,
            aten::hitrecord& rec,
            aten::Intersection& isect) const
        {
            return hit(ctxt, r, t_min, t_max, false, rec, isect);
        }

        void addLight(Light* l)
        {
            m_lights.push_back(l);
        }

        void addImageBasedLight(ImageBasedLight* l)
        {
            if (m_ibl != l) {
                m_ibl = l;

                // TODO
                // Remove light, before adding.
                addLight(l);
            }
        }

        uint32_t lightNum() const
        {
            return (uint32_t)m_lights.size();
        }

        const Light* getLight(uint32_t i) const
        {
            return m_lights[i];
        }

        // TODO
        Light* getLight(uint32_t i)
        {
            return m_lights[i];
        }

        ImageBasedLight* getIBL()
        {
            return m_ibl;
        }

        bool hitLight(
            const aten::context& ctxt,
            const Light* light,
            const aten::vec3& lightPos,
            const aten::ray& r,
            real t_min, real t_max,
            aten::hitrecord& rec);

        static inline AT_DEVICE_API bool hitLight(
            bool isHit,
            aten::LightAttribute attrib,
            const void* lightobj,
            real distToLight,
            real distHitObjToRayOrg,
            const real hitt,
            const void* hitobj)
        {
#if 0
            //auto lightobj = light->object.ptr;

            if (lightobj) {
                // Area Light.
                if (isHit) {
#if 0
                    hitrecord tmpRec;
                    if (lightobj->hit(r, t_min, t_max, tmpRec)) {
                        auto dist2 = squared_length(tmpRec.p - r.org);

                        if (rec->obj == tmpRec.obj
                            && aten::abs(dist2 - rec->t * rec->t) < AT_MATH_EPSILON)
                        {
                            return true;
                        }
                    }
#else
                    //auto distHitObjToRayOrg = (hitp - r.org).length();

                    if (hitobj == lightobj
                        && aten::abs(distHitObjToRayOrg - hitt) <= AT_MATH_EPSILON)
                    {
                        return true;
                    }
#endif
                }
            }

            if (attrib.isInfinite) {
                return !isHit;
            }
            else if (attrib.isSingular) {
                //auto distToLight = (lightPos - r.org).length();

                if (isHit && hitt < distToLight) {
                    // Ray hits something, and the distance to the object is near than the distance to the light.
                    return false;
                }
                else {
                    // Ray don't hit anything, or the distance to the object is far than the distance to the light.
                    return true;
                }
            }

            return false;
#else
            //if (isHit && hitobj == lightobj) {
            if (hitobj == lightobj) {
                return true;
            }

            if (attrib.isInfinite) {
                return !isHit;
            }
            else if (attrib.isSingular) {
                return hitt > distToLight;
            }

            return false;
#endif
        }

        Light* sampleLight(
            const aten::context& ctxt,
            const aten::vec3& org,
            const aten::vec3& nml,
            aten::sampler* sampler,
            real& selectPdf,
            aten::LightSampleResult& sampleRes);

        void drawForGBuffer(
            aten::hitable::FuncPreDraw func,
            std::function<bool(aten::hitable*)> funcIfDraw,
            const aten::context& ctxt) const;

    protected:
        std::vector<aten::hitable*> m_list;

        std::vector<Light*> m_lights;
        ImageBasedLight* m_ibl{ nullptr };
    };

    template <typename ACCEL>
    class AcceleratedScene : public scene {
    public:
        AcceleratedScene()
        {
            aten::accelerator::setInternalAccelType(m_accel.getAccelType());
        }
        virtual ~AcceleratedScene() {}

    public:
        virtual void build(const aten::context& ctxt) override final
        {
            aten::aabb bbox;

            for (const auto& t : m_list) {
                bbox = aten::aabb::merge(bbox, t->getBoundingbox());
            }

            if (!m_list.empty()) {
                // NOTE
                // In "m_accel.build", hitable list will be sorted.
                // To keep order, copy m_list data to another list.
                // This is work around...
                if (m_tmp.empty()) {
                    std::copy(
                        m_list.begin(),
                        m_list.end(),
                        std::back_inserter(m_tmp));
                }

                m_accel.build(ctxt, &m_tmp[0], (uint32_t)m_tmp.size(), &bbox);
            }
        }

        virtual bool hit(
            const aten::context& ctxt,
            const aten::ray& r,
            real t_min, real t_max,
            bool enableLod,
            aten::hitrecord& rec,
            aten::Intersection& isect) const override final
        {
            auto isHit = m_accel.hit(ctxt, r, t_min, t_max, enableLod, isect);

            // TODO
#ifndef __AT_CUDA__
            if (isHit) {
                auto obj = ctxt.getTransformable(isect.objid);
                aten::hitable::evalHitResult(ctxt, obj, r, rec, isect);
            }
#endif

            return isHit;
        }

        ACCEL* getAccel()
        {
            return &m_accel;
        }

    private:
        ACCEL m_accel;
        std::vector<aten::hitable*> m_tmp;
    };
}
