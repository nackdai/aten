#pragma once

#include "defs.h"
#include "types.h"
#include "visualizer/fbo.h"
#include "visualizer/shader.h"
#include "visualizer/GeomDataBuffer.h"
#include "math/mat4.h"
#include "geometry/PolygonObject.h"
#include "scene/host_scene_context.h"
#include "scene/scene.h"
#include "camera/camera.h"

namespace aten {
    class accelerator;

    class RasterizeRenderer {
        static bool s_isInitGlobalVB;

    public:
        RasterizeRenderer() = default;
        ~RasterizeRenderer() = default;

        RasterizeRenderer(const RasterizeRenderer&) = delete;
        RasterizeRenderer(RasterizeRenderer&&) = delete;
        const RasterizeRenderer& operator=(const RasterizeRenderer&) = delete;
        const RasterizeRenderer& operator=(RasterizeRenderer&&) = delete;

    public:
        enum Buffer {
            Color = 0x001,
            Depth = 0x010,
            Sencil = 0x100,
        };

        static void clearBuffer(
            uint32_t clear_buffer_mask,
            aten::vec4& clear_color,
            float clear_depth,
            int32_t clear_stencil);

        static void beginRender(FBO* fbo = nullptr);
        static void beginRender(std::shared_ptr<FBO> fbo)
        {
            beginRender(fbo.get());
        }

        bool init(
            int32_t width, int32_t height,
            std::string_view pathVS,
            std::string_view pathFS);

        bool init(
            int32_t width, int32_t height,
            std::string_view pathVS,
            std::string_view pathGS,
            std::string_view pathFS);

        void release()
        {
            m_boxvb.clear();
            m_vb.clear();
            m_ib.clear();
        }

        void prepareDraw(const camera* cam);

        void drawSceneForGBuffer(
            int32_t frame,
            context& ctxt,
            const scene* scene,
            const camera* cam,
            FBO& fbo,
            shader* exShader = nullptr);

        void drawAABB(
            const camera* cam,
            accelerator* accel);

        void drawAABB(
            const camera* cam,
            const aabb& bbox);

        using FuncSetUniform = std::function<void(shader& shd, const aten::vec3& color, const aten::texture* albedo, int32_t mtrlid)>;

        void drawObject(
            context& ctxt,
            const AT_NAME::PolygonObject& obj,
            const camera* cam,
            bool isWireFrame,
            const mat4& mtx_L2W = mat4::Identity);

        using FuncObjRenderer = std::function<void(const AT_NAME::PolygonObject&)>;

        void drawWithOutsideRenderFunc(
            context& ctxt,
            std::function<void(FuncObjRenderer)> renderFunc,
            const camera* cam,
            bool isWireFrame,
            const mat4& mtx_L2W = mat4::Identity);

        void initBuffer(
            uint32_t vtxStride,
            uint32_t vtxNum,
            const std::vector<uint32_t>& idxNums);

        void draw(
            const context& ctxt,
            const std::vector<vertex>& vtxs,
            const std::vector<std::vector<int32_t>>& idxs,
            const std::vector<material*>& mtrls,
            const camera* cam,
            bool isWireFrame,
            bool updateBuffer);

        void renderSceneDepth(
            context& ctxt,
            const scene* scene,
            const camera* cam);

        void setColor(const vec4& color);

        static void reset()
        {
            s_isInitGlobalVB = false;
        }

        shader& getShader()
        {
            return m_shader;
        }

    private:
        void prepareForDrawAABB(const camera* cam);

    private:
        shader m_shader;
        GeomVertexBuffer m_boxvb;

        mat4 m_mtxPrevW2C;

        int32_t m_width{ 0 };
        int32_t m_height{ 0 };

        bool m_isInitBuffer{ false };
        GeomVertexBuffer m_vb;
        std::vector<GeomIndexBuffer> m_ib;
    };
}
