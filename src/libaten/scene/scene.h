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
            m_list.push_back(s);
        }

        virtual bool hit(
            const aten::context& ctxt,
            const aten::ray& r,
            float t_min, float t_max,
            bool enableLod,
            aten::Intersection& isect,
            aten::HitStopType hit_stop_type = aten::HitStopType::Closest) const = 0;

        bool hit(
            const aten::context& ctxt,
            const aten::ray& r,
            float t_min, float t_max,
            aten::Intersection& isect,
            aten::HitStopType hit_stop_type = aten::HitStopType::Closest) const
        {
            return hit(ctxt, r, t_min, t_max, false, isect, hit_stop_type);
        }

        void addImageBasedLight(
            aten::context& ctxt,
            std::shared_ptr<ImageBasedLight> light)
        {
            if (m_ibl != light) {
                m_ibl = light;

                // TODO
                // Remove light, before adding.
                ctxt.AddLight(light);
            }
        }

        std::shared_ptr<ImageBasedLight> getIBL()
        {
            return m_ibl;
        }

        static inline AT_HOST_DEVICE_API bool hitLight(
            bool is_hit,
            aten::LightAttribute attrib,
            const void* lightobj,
            float distToLight,
            float distHitObjToRayOrg,
            const float hitt,
            const void* hitobj)
        {
#if 0
            //auto lightobj = light->object.ptr;

            if (lightobj) {
                // Area Light.
                if (is_hit) {
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
                return !is_hit;
            }
            else if (attrib.is_singular) {
                //auto distToLight = (lightPos - r.org).length();

                if (is_hit && hitt < distToLight) {
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
            //if (is_hit && hitobj == lightobj) {
            if (hitobj == lightobj) {
                return true;
            }

            if (attrib.isInfinite) {
                return !is_hit;
            }
            else if (attrib.is_singular) {
                return hitt > distToLight;
            }

            return false;
#endif
        }

        void render(
            aten::hitable::FuncPreDraw func,
            std::function<bool(const std::shared_ptr<aten::hitable>&)> funcIfDraw,
            const aten::context& ctxt) const;

        virtual aten::aabb GetBoundingBox() const = 0;

    protected:
        std::vector<std::shared_ptr<aten::hitable>> m_list;

        std::shared_ptr<ImageBasedLight> m_ibl;
    };
}
