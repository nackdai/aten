#pragma once

#include <atomic>

#include "types.h"
#include "material/material.h"
#include "math/mat4.h"
#include "geometry/face.h"
#include "visualizer/GeomDataBuffer.h"
#include "scene/context.h"

namespace AT_NAME
{
    using FuncObjectMeshDraw = std::function<void(const aten::vec3&, const aten::texture*, int32_t)>;

    class objshape : public aten::geombase {
        friend class object;

    public:
        objshape()
        {
            param.type = aten::GeometryType::Polygon;
        }
        virtual ~objshape();

        void build(const aten::context& ctxt);

        void setMaterial(const std::shared_ptr<material>& mtrl)
        {
            m_mtrl = mtrl;
        }

        const std::shared_ptr<material>& getMaterial() const
        {
            return m_mtrl;
        }
        std::shared_ptr<material> getMaterial()
        {
            return m_mtrl;
        }

        void addFace(const std::shared_ptr<face>& f);

        void render(
            aten::hitable::FuncPreDraw func,
            const aten::context& ctxt,
            const aten::mat4& mtxL2W,
            const aten::mat4& mtxPrevL2W,
            int32_t parentId);

        void draw(
            AT_NAME::FuncObjectMeshDraw func,
            const aten::context& ctxt);

        const std::vector<std::shared_ptr<face>>& tris() const
        {
            return faces;
        }

        aten::GeomParameter param;
        aten::aabb m_aabb;

    private:
        std::shared_ptr<material> m_mtrl;
        std::vector<std::shared_ptr<face>> faces;

        aten::GeomIndexBuffer m_ib;

        int32_t m_baseIdx{ INT32_MAX };
        int32_t m_baseTriIdx{ INT32_MAX };
    };
}
