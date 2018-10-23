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
    using FuncObjectMeshDraw = std::function<void(const aten::vec3&, const aten::texture*, int)>;

    class objshape : public aten::geombase {
        friend class object;

    public:
        objshape()
        {
            param.type = aten::GeometryType::Polygon;
        }
        virtual ~objshape();

        void build(const aten::context& ctxt);

        void setMaterial(material* mtrl)
        {
            m_mtrl = mtrl;
        }

        const material* getMaterial() const
        {
            return m_mtrl;
        }
        material* getMaterial()
        {
            return m_mtrl;
        }

        void addFace(face* f);

        void drawForGBuffer(
            aten::hitable::FuncPreDraw func,
            const aten::context& ctxt,
            const aten::mat4& mtxL2W,
            const aten::mat4& mtxPrevL2W,
            int parentId);

        void draw(
            AT_NAME::FuncObjectMeshDraw func,
            const aten::context& ctxt);

        const std::vector<face*>& tris() const
        {
            return faces;
        }

        aten::GeomParameter param;
        aten::aabb m_aabb;

    private:
        material* m_mtrl{ nullptr };
        std::vector<face*> faces;

        aten::GeomIndexBuffer m_ib;

        int m_baseIdx{ INT32_MAX };
        int m_baseTriIdx{ INT32_MAX };
    };
}
