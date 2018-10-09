#pragma once

#include "deformable/MDLFormat.h"
#include "deformable/DeformMesh.h"
#include "deformable/Skeleton.h"
#include "deformable/SkinningVertex.h"
#include "deformable/DeformAnimation.h"
#include "geometry/transformable.h"
#include "visualizer/shader.h"

namespace aten
{
    template<typename T> class instance;

    /** メッシュデータ.
     */
    class deformable : public transformable {
        friend class instance<deformable>;

    public:
        deformable() 
            : m_param(aten::GeometryType::Polygon), transformable(aten::GeometryType::Polygon)
        {}
        ~deformable();

    public:
        bool read(const char* path);

        void release();

        void update(const mat4& mtxL2W);

        void update(
            const mat4& mtxL2W,
            DeformAnimation* anm,
            real time);

        void render(shader* shd);

        void build();

        void getGeometryData(
            std::vector<SkinningVertex>& vtx,
            std::vector<uint32_t>& idx,
            std::vector<aten::PrimitiveParamter>& tris) const;

        const std::vector<mat4>& getMatrices() const;

        virtual void draw(
            aten::hitable::FuncPreDraw func,
            const aten::mat4& mtxL2W,
            const aten::mat4& mtxPrevL2W,
            int parentId,
            uint32_t triOffset) override final;

        bool isEnabledForGPUSkinning() const
        {
            return m_mesh.getDesc().isGPUSkinning;
        }

        GeomMultiVertexBuffer& getVBForGPUSkinning()
        {
            return m_mesh.getVBForGPUSkinning();
        }

        virtual const aten::GeomParameter& getParam() const override final
        {
            return m_param;
        }

        virtual aten::accelerator* getInternalAccelerator() override final
        {
            return m_accel;
        }

        virtual uint32_t getTriangleCount() const override final
        {
            return m_mesh.getTriangleCount();
        }

        virtual bool isDeformable() const override final
        {
            return true;
        }

    private:
        virtual bool hit(
            const ray& r,
            real t_min, real t_max,
            Intersection& isect) const override final
        {
            // Not support.
            AT_ASSERT(false);
            return false;
        }

        virtual void getSamplePosNormalArea(
            aten::hitable::SamplePosNormalPdfResult* result,
            const mat4& mtxL2W,
            sampler* sampler) const override final
        {
            // Not support.
            AT_ASSERT(false);
        }

        virtual void evalHitResult(
            const ray& r,
            const mat4& mtxL2W,
            hitrecord& rec,
            const Intersection& isect) const override final
        {
            // Not support.
            AT_ASSERT(false);
        }

    private:
        DeformMesh m_mesh;

        GeomParameter m_param;
        aten::accelerator* m_accel{ nullptr };

        // TODO
        Skeleton m_skl;
        SkeletonController m_sklController;
    };

    //////////////////////////////////////////////////////////////

    class camera;
    class DeformMeshReadHelper;

    // For debug.
    class DeformableRenderer {
        friend class deformable;

    private:
        DeformableRenderer();
        ~DeformableRenderer();

    public:
        static bool init(
            int width, int height,
            const char* pathVS,
            const char* pathFS);

        static void render(
            const camera* cam,
            deformable* mdl);

    private:
        static void initDeformMeshReadHelper(DeformMeshReadHelper* helper);

    private:
        static shader s_shd;
    };
}
