#pragma once

#include <vector>
#include <algorithm>
#include <iterator>

#include "accelerator/accelerator.h"
#include "accelerator/bvh.h"
#include "light/light.h"
#include "light/ibl.h"
#include "scene/host_scene_context.h"
#include "scene/scene.h"

namespace aten {
    template <class ACCEL>
    class AcceleratedScene : public scene {
    public:
        AcceleratedScene()
        {
            aten::accelerator::setInternalAccelType(m_accel.getAccelType());
        }
        virtual ~AcceleratedScene() {}

    public:
        virtual void build(const aten::context& ctxt) final
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
                    for (auto& i : m_list) {
                        m_tmp.push_back(i.get());
                    }
                }

                m_accel.build(ctxt, &m_tmp[0], (uint32_t)m_tmp.size(), &bbox);
            }
        }

        virtual bool hit(
            const aten::context& ctxt,
            const aten::ray& r,
            float t_min, float t_max,
            bool enableLod,
            aten::Intersection& isect,
            aten::HitStopType hit_stop_type = aten::HitStopType::Closest) const override final
        {
            isect.t = t_max;
            auto isHit = m_accel.HitWithLod(ctxt, r, t_min, t_max, enableLod, isect, hit_stop_type);

            return isHit;
        }

        ACCEL* getAccel()
        {
            return &m_accel;
        }

        aten::aabb GetBoundingBox() const override
        {
            return m_accel.GetBoundingBox();
        }

    private:
        ACCEL m_accel;
        std::vector<aten::hitable*> m_tmp;
    };
}
