#pragma once

#include <vector>
#include <algorithm>
#include <iterator>

#include "accelerator/accelerator.h"
#include "accelerator/bvh.h"
#include "light/light.h"
#include "light/ibl.h"
#include "scene/host_scene_context.h"

namespace AT_NAME {
    class scene {
    public:
        scene() = default;
        virtual ~scene() {}

    public:
        virtual void build(const aten::context& ctxt)
        {}

        void add(const std::shared_ptr<aten::hitable>& s)
        {
            m_aabb.merge(s->getBoundingbox());

            m_list.push_back(s);
        }

        virtual bool hit(
            const aten::context& ctxt,
            const aten::ray& r,
            real t_min, real t_max,
            bool enableLod,
            aten::Intersection& isect) const = 0;

        bool hit(
            const aten::context& ctxt,
            const aten::ray& r,
            real t_min, real t_max,
            aten::Intersection& isect) const
        {
            return hit(ctxt, r, t_min, t_max, false, isect);
        }

        void addImageBasedLight(
            aten::context& ctxt,
            std::shared_ptr<ImageBasedLight> light)
        {
            if (m_ibl != light) {
                m_ibl = light;

                // TODO
                // Remove light, before adding.
                ctxt.add_light(light);
            }
        }

        std::shared_ptr<ImageBasedLight> getIBL()
        {
            return m_ibl;
        }

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

        std::shared_ptr<Light> sampleLight(
            const aten::context& ctxt,
            const aten::vec3& org,
            const aten::vec3& nml,
            aten::sampler* sampler,
            real& selectPdf,
            aten::LightSampleResult& sampleRes);

        std::shared_ptr<Light> sampleLightWithReservoir(
            const aten::context& ctxt,
            const aten::vec3& org,
            const aten::vec3& nml,
            std::function<aten::vec3(const aten::vec3&)> compute_brdf,
            aten::sampler* sampler,
            real& selectPdf,
            aten::LightSampleResult& sampleRes);

        void render(
            aten::hitable::FuncPreDraw func,
            std::function<bool(const std::shared_ptr<aten::hitable>&)> funcIfDraw,
            const aten::context& ctxt) const;

        const aten::aabb& getBoundingBox() const
        {
            return m_aabb;
        }

    protected:
        std::vector<std::shared_ptr<aten::hitable>> m_list;

        std::shared_ptr<ImageBasedLight> m_ibl;

        aten::aabb m_aabb;
    };
}
