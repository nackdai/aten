#pragma once

#include <memory>

#include "deformable/MDLFormat.h"
#include "deformable/DeformMesh.h"
#include "deformable/Skeleton.h"
#include "deformable/SkinningVertex.h"
#include "deformable/DeformAnimation.h"
#include "geometry/transformable.h"
#include "visualizer/shader.h"
#include "scene/host_scene_context.h"
#include "camera/camera.h"

namespace aten
{
    class DeformableRenderer;

    /** メッシュデータ.
     */
    class deformable : public transformable {
        friend class DeformableRenderer;
        friend class TransformableFactory;

    public:
        deformable()
            : transformable(aten::ObjectType::Polygons)
        {}

        ~deformable() = default;

        bool read(std::string_view path);

        void initGLResources(shader* shd);
        void initGLResourcesWithDeformableRenderer(DeformableRenderer& renderer);

        void release();

        void update(const mat4& mtx_L2W);

        void update(
            const mat4& mtx_L2W,
            float time,
            DeformAnimation* anm);

        void build();

        void getGeometryData(
            const context& ctxt,
            std::vector<SkinningVertex>& vtx,
            std::vector<uint32_t>& idx,
            std::vector<aten::TriangleParameter>& tris) const;

        const std::vector<mat4>& getMatrices() const;

        virtual void render(
            aten::hitable::FuncPreDraw func,
            const context& ctxt,
            const aten::mat4& mtx_L2W,
            const aten::mat4& mtx_prev_L2W,
            int32_t parentId,
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
            return m_accel.get();
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
            float t_min, float t_max,
            Intersection& isect) const override final
        {
            // Not support.
            AT_ASSERT(false);
            return false;
        }

    private:
        void render(
            const context& ctxt,
            shader* shd);

    private:
        DeformMesh m_mesh;

        std::shared_ptr<accelerator> m_accel;

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
        DeformableRenderer() = default;
        ~DeformableRenderer() = default;

        DeformableRenderer(const DeformableRenderer& rhs) = delete;
        const DeformableRenderer& operator=(const DeformableRenderer& rhs) = delete;

    public:
        bool init(
            int32_t width, int32_t height,
            std::string_view pathVS,
            std::string_view pathFS);

        void render(
            const context& ctxt,
            const Camera* cam,
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
