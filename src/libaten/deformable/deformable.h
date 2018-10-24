#pragma once

#include "deformable/MDLFormat.h"
#include "deformable/DeformMesh.h"
#include "deformable/Skeleton.h"
#include "deformable/SkinningVertex.h"
#include "deformable/DeformAnimation.h"
#include "geometry/transformable.h"
#include "visualizer/shader.h"
#include "scene/context.h"
#include "camera/camera.h"

namespace aten
{
    class DeformableRenderer;

    /** メッシュデータ.
     */
    class deformable : public transformable {
        friend class DeformableRenderer;
        friend class TransformableFactory;

    private:
        deformable() 
            : transformable(aten::GeometryType::Polygon)
        {}

    public:
        ~deformable();

    public:
        bool read(const char* path);

        void initGLResources(shader* shd);
        void initGLResourcesWithDeformableRenderer(DeformableRenderer& renderer);

        void release();

        void update(const mat4& mtxL2W);

        void update(
            const mat4& mtxL2W,
            real time,
            DeformAnimation* anm);

        void build();

        void getGeometryData(
            const context& ctxt,
            std::vector<SkinningVertex>& vtx,
            std::vector<uint32_t>& idx,
            std::vector<aten::PrimitiveParamter>& tris) const;

        const std::vector<mat4>& getMatrices() const;

        virtual void drawForGBuffer(
            aten::hitable::FuncPreDraw func,
            const context& ctxt,
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

        virtual bool hit(
            const context& ctxt,
            const ray& r,
            real t_min, real t_max,
            Intersection& isect) const override final
        {
            // Not support.
            AT_ASSERT(false);
            return false;
        }

        virtual void getSamplePosNormalArea(
            const context& ctxt,
            aten::hitable::SamplePosNormalPdfResult* result,
            const mat4& mtxL2W,
            sampler* sampler) const override final
        {
            // Not support.
            AT_ASSERT(false);
        }

        virtual void evalHitResult(
            const context& ctxt,
            const ray& r,
            const mat4& mtxL2W,
            hitrecord& rec,
            const Intersection& isect) const override final
        {
            // Not support.
            AT_ASSERT(false);
        }

    private:
        void render(
            const context& ctxt,
            shader* shd);

    private:
        DeformMesh m_mesh;

        aten::accelerator* m_accel{ nullptr };

        bool m_isInitializedToRender{ false };

        // TODO
        Skeleton m_skl;
        SkeletonController m_sklController;
    };

    //////////////////////////////////////////////////////////////

    // For debug.
    class DeformableRenderer {
        friend class deformable;

    public:
        DeformableRenderer() {}
        ~DeformableRenderer() {}
            
    private:
        DeformableRenderer(const DeformableRenderer& rhs) = delete;
        const DeformableRenderer& operator=(const DeformableRenderer& rhs) = delete;

    public:
        bool init(
            int width, int height,
            const char* pathVS,
            const char* pathFS);

        void render(
            const context& ctxt,
            const camera* cam,
            deformable* mdl);

    private:
        shader* getShader()
        {
            return &m_shd;
        }

    private:
        shader m_shd;
    };
}
