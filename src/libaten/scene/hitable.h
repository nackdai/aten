#pragma once

#include <vector>
#include <functional>

#include "types.h"
#include "math/aabb.h"
#include "math/vec3.h"
#include "sampler/sampler.h"
#include "scene/hit_parameter.h"

namespace aten {
    class context;
    class hitable;
    class accelerator;

    using NotifyChanged = std::function<void(hitable*)>;

    class hitable {
    public:
        hitable(std::string_view name = {})
        {
            if (!name.empty()) {
                m_name = name;
            }
        }
        virtual ~hitable() = default;

    public:
        virtual bool hit(
            const context& ctxt,
            const ray& r,
            real t_min, real t_max,
            Intersection& isect) const = 0;

        virtual const aabb& getBoundingbox() const
        {
            return m_aabb;
        }
        void setBoundingBox(const aabb& bbox)
        {
            m_aabb = bbox;
        }

        virtual aabb getTransformedBoundingBox() const
        {
            return m_aabb;
        }

        virtual const hitable* getHasObject() const
        {
            return nullptr;
        }

        virtual const hitable* getHasSecondObject() const
        {
            return nullptr;
        }

        bool isInstance() const
        {
            return (getHasObject() != nullptr);
        }

        virtual int32_t GetMeshId() const
        {
            return -1;
        }

        virtual accelerator* getInternalAccelerator();

        using FuncPreDraw = std::function<void(const aten::mat4& mtx_L2W, const aten::mat4& mtx_prev_L2W, int32_t parentId, int32_t basePrimId)>;

        virtual void render(
            FuncPreDraw func,
            const context& ctxt,
            const aten::mat4& mtx_L2W,
            const aten::mat4& mtx_prev_L2W,
            int32_t parentId,
            uint32_t triOffset)
        {
            // For rasterize rendering.
            AT_ASSERT(false);
        }

        using FuncDrawAABB = std::function<void(const aten::mat4&)>;

        virtual void drawAABB(
            FuncDrawAABB func,
            const aten::mat4& mtx_L2W)
        {
            // For debug rendering.
            AT_ASSERT(false);
        }

        void setFuncNotifyChanged(NotifyChanged onNotifyChanged)
        {
            m_onNotifyChanged = onNotifyChanged;
        }

        virtual void update()
        {
            // Nothing is done...
        }

        virtual uint32_t getTriangleCount() const
        {
            return 0;
        }

        virtual bool isDeformable() const
        {
            return false;
        }

        void setName(std::string_view name)
        {
            m_name = name;
        }
        const char* name()
        {
            return m_name.c_str();
        }

    protected:
        void onNotifyChanged()
        {
            if (m_onNotifyChanged) {
                m_onNotifyChanged(this);
            }
        }

    private:
        std::string m_name;
        aabb m_aabb;

        NotifyChanged m_onNotifyChanged;
    };
}
