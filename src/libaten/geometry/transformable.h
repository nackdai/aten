#pragma once

#include <vector>

#include "types.h"
#include "scene/hitable.h"
#include "math/mat4.h"
#include "geometry/geomparam.h"
#include "misc/datalist.h"

namespace aten
{
    class transformable : public hitable {
        friend class context;

    protected:
        transformable();
        virtual ~transformable();

        transformable(GeometryType type);

    public:
        virtual void getSamplePosNormalArea(
            const context& ctxt,
            aten::hitable::SamplePosNormalPdfResult* result,
            const mat4& mtxL2W,
            sampler* sampler) const = 0;

        virtual void evalHitResult(
            const context& ctxt,
            const ray& r,
            const mat4& mtxL2W,
            hitrecord& rec,
            const Intersection& isect) const = 0;

        GeometryType getType() const
        {
            return m_param.type;
        }

        const GeomParameter& getParam() const
        {
            return m_param;
        }

        GeomParameter& getParam()
        {
            return m_param;
        }

        virtual void getMatrices(
            aten::mat4& mtxL2W,
            aten::mat4& mtxW2L) const
        {
            mtxL2W.identity();
            mtxW2L.identity();
        }

        int id() const
        {
            return m_id;
        }

        virtual void collectTriangles(std::vector<aten::PrimitiveParamter>& triangles) const
        {
            // Nothing is done...
        }

    private:
        static void resetIdWhenAnyTransformableLeave(aten::transformable* obj);

        void addToDataList(aten::DataList<aten::transformable>& list)
        {
            list.add(&m_listItem);
            m_id = m_listItem.currentIndex();
        }

    protected:
        int m_id{ -1 };

        GeomParameter m_param;

        DataList<transformable>::ListItem m_listItem;
    };
}
