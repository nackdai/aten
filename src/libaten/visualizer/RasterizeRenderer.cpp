#include "visualizer/RasterizeRenderer.h"
#include "visualizer/atengl.h"
#include "scene/scene.h"
#include "camera/camera.h"
#include "math/mat4.h"
#include "accelerator/accelerator.h"
#include "sampler/cmj.h"

namespace aten
{
    void RasterizeRenderer::clearBuffer(
        uint32_t clear_buffer_mask,
        aten::vec4 &clear_color,
        float clear_depth,
        int32_t clear_stencil)
    {
        GLbitfield clear_mask = 0;

        if (clear_buffer_mask & static_cast<uint32_t>(Buffer::Color))
        {
            clear_mask |= GL_COLOR_BUFFER_BIT;
            CALL_GL_API(::glClearColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a));
        }
        if (clear_buffer_mask & static_cast<uint32_t>(Buffer::Depth))
        {
            clear_mask |= GL_DEPTH_BUFFER_BIT;
            CALL_GL_API(::glClearDepthf(clear_depth));
        }
        if (clear_buffer_mask & static_cast<uint32_t>(Buffer::Sencil))
        {
            clear_mask |= GL_STENCIL_BUFFER_BIT;
            CALL_GL_API(::glClearStencil(clear_stencil));
        }

        AT_ASSERT(clear_mask > 0);

        CALL_GL_API(::glClear(clear_mask));
    }

    void RasterizeRenderer::beginRender(FBO *fbo /*= nullptr*/)
    {
        if (fbo)
        {
            AT_ASSERT(fbo->IsValid());
            fbo->BindFBO();
        }
        else
        {
            // Set default frame buffer.
            CALL_GL_API(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
        }
    }

    bool RasterizeRenderer::init(
        int32_t width, int32_t height,
        std::string_view pathVS,
        std::string_view pathFS)
    {
        width_ = width;
        height_ = height;

        return m_shader.init(width, height, pathVS, pathFS);
    }

    bool RasterizeRenderer::init(
        int32_t width, int32_t height,
        std::string_view pathVS,
        std::string_view pathGS,
        std::string_view pathFS)
    {
        width_ = width;
        height_ = height;

        return m_shader.init(width, height, pathVS, pathGS, pathFS);
    }

    void RasterizeRenderer::prepareDraw(const camera *cam)
    {
        auto camparam = cam->param();

        // TODO
        camparam.znear = float(0.1);
        camparam.zfar = float(10000.0);

        mat4 mtx_W2V;
        mat4 mtx_V2C;

        mtx_W2V.lookat(
            camparam.origin,
            camparam.center,
            camparam.up);

        mtx_V2C.perspective(
            camparam.znear,
            camparam.zfar,
            camparam.vfov,
            camparam.aspect);

        aten::mat4 mtx_W2C = mtx_V2C * mtx_W2V;

        m_shader.prepareRender(nullptr, false);

        auto hMtxW2C = m_shader.getHandle("mtx_W2C");
        CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, (const GLfloat *)&mtx_W2C.a[0]));
    }

    void RasterizeRenderer::drawSceneForGBuffer(
        int32_t frame,
        context &ctxt,
        const scene *scene,
        const camera *cam,
        FBO &fbo,
        shader *exShader /*= nullptr*/)
    {
        AT_ASSERT(scene);
        AT_ASSERT(cam);

        auto camparam = cam->param();

        // TODO
        camparam.znear = float(0.1);
        camparam.zfar = float(10000.0);

        mat4 mtx_W2V;
        mat4 mtx_V2C;

        mtx_W2V.lookat(
            camparam.origin,
            camparam.center,
            camparam.up);

        mtx_V2C.perspective(
            camparam.znear,
            camparam.zfar,
            camparam.vfov,
            camparam.aspect);

        aten::mat4 mtx_W2C = mtx_V2C * mtx_W2V;

        // TODO
        // For TAA.
        {
            CMJ sampler;
            auto rnd = getRandom(frame);
            auto scramble = rnd * 0x1fe3434f * ((frame + 331 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
            sampler.init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 4 + 300, scramble);

            auto smpl = sampler.nextSample2D();

            aten::mat4 mtxOffset;
            mtxOffset.asTrans(aten::vec3(
                smpl.x / width_,
                smpl.y / height_,
                float(0)));

            mtx_W2C = mtxOffset * mtx_W2C;
        }

        if (m_mtxPrevW2C.isIdentity())
        {
            m_mtxPrevW2C = mtx_W2C;
        }

        m_shader.prepareRender(nullptr, false);

        auto hMtxW2C = m_shader.getHandle("mtx_W2C");
        CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, (const GLfloat *)&mtx_W2C.a[0]));

        auto hPrevMtxW2C = m_shader.getHandle("mtxPrevW2C");
        CALL_GL_API(::glUniformMatrix4fv(hPrevMtxW2C, 1, GL_TRUE, (const GLfloat *)&m_mtxPrevW2C.a[0]));

        AT_ASSERT(fbo.IsValid());
        fbo.BindFBO(true);

        {
            CALL_GL_API(::glEnable(GL_DEPTH_TEST));
            // CALL_GL_API(::glEnable(GL_CULL_FACE));
        }

        // Clear buffer
        float clr[] = {-1.0f, -1.0f, -1.0f, -1.0f};
        CALL_GL_API(glClearNamedFramebufferfv(fbo.GetGLHandle(), GL_COLOR, 0, clr));
        CALL_GL_API(glClearNamedFramebufferfv(fbo.GetGLHandle(), GL_COLOR, 1, clr));

        aten::vec4 tmp_clear_color;
        clearBuffer(
            Buffer::Depth | Buffer::Sencil,
            tmp_clear_color, 1.0f, 0);

        ctxt.build();

        // For object (which means "not" deformable).
        scene->render([&](const aten::mat4 &mtx_L2W, const aten::mat4 &mtx_prev_L2W, int32_t objid, int32_t primid)
                      {
            auto hMtxL2W = m_shader.getHandle("mtx_L2W");
            CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, (const GLfloat*)&mtx_L2W.a[0]));

            auto hPrevMtxL2W = m_shader.getHandle("mtx_prev_L2W");
            CALL_GL_API(::glUniformMatrix4fv(hPrevMtxL2W, 1, GL_TRUE, (const GLfloat*)&mtx_prev_L2W.a[0]));

            auto hObjId = m_shader.getHandle("objid");
            CALL_GL_API(::glUniform1i(hObjId, objid));

            auto hPrimId = m_shader.getHandle("primid");
            CALL_GL_API(::glUniform1i(hPrimId, primid)); },
                      [](const std::shared_ptr<hitable> &target)
                      {
                          return !target->isDeformable();
                      },
                      ctxt);

        // For deformable.
        if (exShader)
        {
            exShader->prepareRender(nullptr, false);

            hMtxW2C = exShader->getHandle("mtx_W2C");
            CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, (const GLfloat *)&mtx_W2C.a[0]));

            hPrevMtxW2C = exShader->getHandle("mtxPrevW2C");
            CALL_GL_API(::glUniformMatrix4fv(hPrevMtxW2C, 1, GL_TRUE, (const GLfloat *)&m_mtxPrevW2C.a[0]));

            scene->render([&](const aten::mat4 &mtx_L2W, const aten::mat4 &mtx_prev_L2W, int32_t objid, int32_t primid)
                          {
                auto hMtxL2W = exShader->getHandle("mtx_L2W");
                CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, (const GLfloat*)&mtx_L2W.a[0]));

                auto hPrevMtxL2W = exShader->getHandle("mtx_prev_L2W");
                CALL_GL_API(::glUniformMatrix4fv(hPrevMtxL2W, 1, GL_TRUE, (const GLfloat*)&mtx_prev_L2W.a[0]));

                auto hObjId = exShader->getHandle("objid");
                CALL_GL_API(::glUniform1i(hObjId, objid));

                auto hPrimId = exShader->getHandle("primid");
                CALL_GL_API(::glUniform1i(hPrimId, primid)); },
                          [](const std::shared_ptr<hitable> &target)
                          {
                              return target->isDeformable();
                          },
                          ctxt);
        }

        // Set default frame buffer.
        CALL_GL_API(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));

        m_mtxPrevW2C = mtx_W2C;
    }

    static const vertex boxvtx[] = {
        // 0
        {{0, 0, 0, 1}, {0, 0, 0}, {1, 0, 0}},
        {{1, 0, 0, 1}, {0, 0, 0}, {1, 0, 0}},

        // 1
        {{0, 0, 0, 1}, {0, 0, 0}, {1, 0, 0}},
        {{0, 1, 0, 1}, {0, 0, 0}, {1, 0, 0}},

        // 2
        {{0, 0, 0, 1}, {0, 0, 0}, {1, 0, 0}},
        {{0, 0, 1, 1}, {0, 0, 0}, {1, 0, 0}},

        // 3
        {{1, 0, 0, 1}, {0, 0, 0}, {1, 0, 0}},
        {{1, 0, 1, 1}, {0, 0, 0}, {1, 0, 0}},

        // 4
        {{1, 0, 0, 1}, {0, 0, 0}, {1, 0, 0}},
        {{1, 1, 0, 1}, {0, 0, 0}, {1, 0, 0}},

        // 5
        {{0, 1, 0, 1}, {0, 0, 0}, {1, 0, 0}},
        {{1, 1, 0, 1}, {0, 0, 0}, {1, 0, 0}},

        // 6
        {{0, 1, 0, 1}, {0, 0, 0}, {1, 0, 0}},
        {{0, 1, 1, 1}, {0, 0, 0}, {1, 0, 0}},

        // 7
        {{0, 0, 1, 1}, {0, 0, 0}, {1, 0, 0}},
        {{1, 0, 1, 1}, {0, 0, 0}, {1, 0, 0}},

        // 8
        {{0, 0, 1, 1}, {0, 0, 0}, {1, 0, 0}},
        {{0, 1, 1, 1}, {0, 0, 0}, {1, 0, 0}},

        // 9
        {{1, 0, 1, 1}, {0, 0, 0}, {1, 0, 0}},
        {{1, 1, 1, 1}, {0, 0, 0}, {1, 0, 0}},

        // 10
        {{1, 1, 0, 1}, {0, 0, 0}, {1, 0, 0}},
        {{1, 1, 1, 1}, {0, 0, 0}, {1, 0, 0}},

        {{0, 1, 1, 1}, {0, 0, 0}, {1, 0, 0}},
        {{1, 1, 1, 1}, {0, 0, 0}, {1, 0, 0}},
    };

    void RasterizeRenderer::drawAABB(
        const camera *cam,
        accelerator *accel)
    {
        prepareForDrawAABB(cam);

        auto hMtxL2W = m_shader.getHandle("mtx_L2W");

        accel->drawAABB([&](const aten::mat4 &mtx_L2W)
                        {
            // Draw.
            CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, (const GLfloat*)&mtx_L2W.a[0]));
            m_boxvb.draw(aten::Primitive::Lines, 0, 12); },
                        aten::mat4::Identity);
    }

    void RasterizeRenderer::drawAABB(
        const camera *cam,
        const aabb &bbox)
    {
        prepareForDrawAABB(cam);

        auto hMtxL2W = m_shader.getHandle("mtx_L2W");

        aten::mat4 mtxScale;
        mtxScale.asScale(bbox.size());

        aten::mat4 mtxTrans;
        mtxTrans.asTrans(bbox.minPos());

        aten::mat4 mtx_L2W = mtxTrans * mtxScale;

        // Draw.
        CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, (const GLfloat *)&mtx_L2W.a[0]));
        m_boxvb.draw(aten::Primitive::Lines, 0, 12);
    }

    void RasterizeRenderer::prepareForDrawAABB(const camera *cam)
    {
        // Initialize vb.
        if (!m_boxvb.isInitialized())
        {
            m_boxvb.init(
                sizeof(aten::vertex),
                AT_COUNTOF(boxvtx),
                0,
                (void *)boxvtx);
        }

        m_shader.prepareRender(nullptr, false);

        auto camparam = cam->param();

        // TODO
        camparam.znear = float(0.1);
        camparam.zfar = float(10000.0);

        mat4 mtx_W2V;
        mat4 mtx_V2C;

        mtx_W2V.lookat(
            camparam.origin,
            camparam.center,
            camparam.up);

        mtx_V2C.perspective(
            camparam.znear,
            camparam.zfar,
            camparam.vfov,
            camparam.aspect);

        aten::mat4 mtx_W2C = mtx_V2C * mtx_W2V;

        auto hMtxW2C = m_shader.getHandle("mtx_W2C");
        CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, (const GLfloat *)&mtx_W2C.a[0]));
    }

    void RasterizeRenderer::drawObject(
        context &ctxt,
        const AT_NAME::PolygonObject &obj,
        const camera *cam,
        bool isWireFrame,
        const mat4 &mtx_L2W)
    {
        auto camparam = cam->param();

        // TODO
        camparam.znear = float(0.1);
        camparam.zfar = float(10000.0);

        mat4 mtx_W2V;
        mat4 mtx_V2C;

        mtx_W2V.lookat(
            camparam.origin,
            camparam.center,
            camparam.up);

        mtx_V2C.perspective(
            camparam.znear,
            camparam.zfar,
            camparam.vfov,
            camparam.aspect);

        aten::mat4 mtx_W2C = mtx_V2C * mtx_W2V;

        m_shader.prepareRender(nullptr, false);

        // Not modify local to world matrix...
        auto hMtxL2W = m_shader.getHandle("mtx_L2W");
        CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, (const GLfloat *)&mtx_L2W.a[0]));

        auto hMtxW2C = m_shader.getHandle("mtx_W2C");
        CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, (const GLfloat *)&mtx_W2C.a[0]));

        if (isWireFrame)
        {
            CALL_GL_API(::glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));
        }
        else
        {
            CALL_GL_API(::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
        }

        CALL_GL_API(::glEnable(GL_DEPTH_TEST));
        CALL_GL_API(::glEnable(GL_CULL_FACE));

        ctxt.build();

        auto hHasAlbedo = m_shader.getHandle("hasAlbedo");
        auto hColor = m_shader.getHandle("color");
        auto hMtrlId = m_shader.getHandle("materialId");

        obj.draw([&](const aten::vec3 &color, const aten::texture *albedo, int32_t mtrlid)
                 {
            if (albedo) {
                albedo->bindAsGLTexture(0, &m_shader);
                CALL_GL_API(::glUniform1i(hHasAlbedo, true));
            }
            else {
                CALL_GL_API(::glUniform1i(hHasAlbedo, false));
            }

            CALL_GL_API(::glUniform4f(hColor, color.x, color.y, color.z, 1.0f));

            if (hMtrlId >= 0) {
                CALL_GL_API(::glUniform1i(hMtrlId, mtrlid));
            } },
                 ctxt);

        // 戻す.
        CALL_GL_API(::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
    }

    void RasterizeRenderer::drawWithOutsideRenderFunc(
        context &ctxt,
        std::function<void(FuncObjRenderer)> renderFunc,
        const camera *cam,
        bool isWireFrame,
        const mat4 &mtx_L2W /*= mat4::Identity*/)
    {
        auto camparam = cam->param();

        // TODO
        camparam.znear = float(0.1);
        camparam.zfar = float(10000.0);

        mat4 mtx_W2V;
        mat4 mtx_V2C;

        mtx_W2V.lookat(
            camparam.origin,
            camparam.center,
            camparam.up);

        mtx_V2C.perspective(
            camparam.znear,
            camparam.zfar,
            camparam.vfov,
            camparam.aspect);

        aten::mat4 mtx_W2C = mtx_V2C * mtx_W2V;

        m_shader.prepareRender(nullptr, false);

        // Not modify local to world matrix...
        auto hMtxL2W = m_shader.getHandle("mtx_L2W");
        CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, (const GLfloat *)&mtx_L2W.a[0]));

        auto hMtxW2C = m_shader.getHandle("mtx_W2C");
        CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, (const GLfloat *)&mtx_W2C.a[0]));

        if (isWireFrame)
        {
            CALL_GL_API(::glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));
        }
        else
        {
            CALL_GL_API(::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
        }

        CALL_GL_API(::glEnable(GL_DEPTH_TEST));
        CALL_GL_API(::glEnable(GL_CULL_FACE));

        ctxt.build();

        auto hHasAlbedo = m_shader.getHandle("hasAlbedo");
        auto hColor = m_shader.getHandle("color");
        auto hMtrlId = m_shader.getHandle("materialId");

        renderFunc([&](const AT_NAME::PolygonObject&obj)
                   { obj.draw([&](const aten::vec3 &color, const aten::texture *albedo, int32_t mtrlid)
                              {
                if (hHasAlbedo >= 0) {
                    if (albedo) {
                        albedo->bindAsGLTexture(0, &m_shader);
                        CALL_GL_API(::glUniform1i(hHasAlbedo, true));
                    }
                    else {
                        CALL_GL_API(::glUniform1i(hHasAlbedo, false));
                    }
                }

                CALL_GL_API(::glUniform4f(hColor, color.x, color.y, color.z, 1.0f));

                if (hMtrlId >= 0) {
                    CALL_GL_API(::glUniform1i(hMtrlId, mtrlid));
                } },
                              ctxt); });

        // 戻す.
        CALL_GL_API(::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
    }

    void RasterizeRenderer::initBuffer(
        uint32_t vtxStride,
        uint32_t vtxNum,
        const std::vector<uint32_t> &idxNums)
    {
        if (!m_isInitBuffer)
        {
            vertex_buffer_.init(
                sizeof(vertex),
                vtxNum,
                0,
                nullptr);

            m_ib.resize(idxNums.size());

            for (int32_t i = 0; i < idxNums.size(); i++)
            {
                auto num = idxNums[i];
                m_ib[i].init(num, nullptr);
            }

            m_isInitBuffer = true;
        }
    }

    void RasterizeRenderer::draw(
        const context &ctxt,
        const std::vector<vertex> &vtxs,
        const std::vector<std::vector<int32_t>> &idxs,
        const std::vector<material *> &mtrls,
        const camera *cam,
        bool isWireFrame,
        bool updateBuffer)
    {
        AT_ASSERT(idxs.size() == mtrls.size());

        auto camparam = cam->param();

        // TODO
        camparam.znear = float(0.1);
        camparam.zfar = float(10000.0);

        mat4 mtx_W2V;
        mat4 mtx_V2C;

        mtx_W2V.lookat(
            camparam.origin,
            camparam.center,
            camparam.up);

        mtx_V2C.perspective(
            camparam.znear,
            camparam.zfar,
            camparam.vfov,
            camparam.aspect);

        aten::mat4 mtx_W2C = mtx_V2C * mtx_W2V;

        m_shader.prepareRender(nullptr, false);

        // Not modify local to world matrix...
        auto hMtxL2W = m_shader.getHandle("mtx_L2W");
        CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, (const GLfloat *)&mat4::Identity.a[0]));

        auto hMtxW2C = m_shader.getHandle("mtx_W2C");
        CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, (const GLfloat *)&mtx_W2C.a[0]));

        // Set default frame buffer.
        CALL_GL_API(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));

        if (isWireFrame)
        {
            CALL_GL_API(::glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));
        }
        else
        {
            CALL_GL_API(::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
        }

        CALL_GL_API(::glEnable(GL_DEPTH_TEST));
        CALL_GL_API(::glEnable(GL_CULL_FACE));

        if (!m_isInitBuffer)
        {
            vertex_buffer_.init(
                sizeof(vertex),
                static_cast<uint32_t>(vtxs.size()),
                0,
                vtxs.data());

            m_ib.resize(idxs.size());

            for (size_t i = 0; i < idxs.size(); i++)
            {
                if (!idxs[i].empty())
                {
                    m_ib[i].init(static_cast<uint32_t>(idxs[i].size()), idxs[i].data());
                }
            }

            m_isInitBuffer = true;
        }
        else if (updateBuffer)
        {
            // TODO
            // 最初に最大数バッファをアロケートしておかないといけない...

            vertex_buffer_.update(vtxs.size(), vtxs.data());

            for (size_t i = 0; i < idxs.size(); i++)
            {
                if (!idxs[i].empty())
                {
                    m_ib[i].update(static_cast<uint32_t>(idxs[i].size()), idxs[i].data());
                }
            }
        }

        auto hHasAlbedo = m_shader.getHandle("hasAlbedo");

        for (int32_t i = 0; i < m_ib.size(); i++)
        {
            auto triNum = static_cast<int32_t>(idxs[i].size() / 3);

            if (triNum > 0)
            {
                auto m = mtrls[i];

                int32_t albedoTexId = m ? m->param().albedoMap : -1;
                const auto &albedo = albedoTexId >= 0 ? ctxt.GetTexture(albedoTexId) : nullptr;

                if (albedo)
                {
                    albedo->bindAsGLTexture(0, &m_shader);
                    CALL_GL_API(::glUniform1i(hHasAlbedo, true));
                }
                else
                {
                    CALL_GL_API(::glUniform1i(hHasAlbedo, false));
                }

                m_ib[i].draw(vertex_buffer_, aten::Primitive::Triangles, 0, triNum);
            }
        }

        // 戻す.
        CALL_GL_API(::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
    }

    void RasterizeRenderer::renderSceneDepth(
        context &ctxt,
        const scene *scene,
        const camera *cam)
    {
        auto camparam = cam->param();

        // TODO
        camparam.znear = float(0.1);
        camparam.zfar = float(10000.0);

        mat4 mtx_W2V;
        mat4 mtx_V2C;

        mtx_W2V.lookat(
            camparam.origin,
            camparam.center,
            camparam.up);

        mtx_V2C.perspective(
            camparam.znear,
            camparam.zfar,
            camparam.vfov,
            camparam.aspect);

        aten::mat4 mtx_W2C = mtx_V2C * mtx_W2V;

        m_shader.prepareRender(nullptr, false);

        auto hMtxW2C = m_shader.getHandle("mtx_W2C");
        CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, (const GLfloat *)&mtx_W2C.a[0]));

        // Set default frame buffer.
        CALL_GL_API(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));

        CALL_GL_API(::glEnable(GL_DEPTH_TEST));

        // No need color rendering.
        CALL_GL_API(::glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE));

        aten::vec4 tmp_clear_color;
        clearBuffer(
            Buffer::Depth | Buffer::Sencil,
            tmp_clear_color, 1.0f, 0);

        ctxt.build();

        scene->render(
            [&](const aten::mat4 &mtx_L2W, const aten::mat4 &mtx_prev_L2W, int32_t objid, int32_t primid)
            {
                auto hMtxL2W = m_shader.getHandle("mtx_L2W");
                CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, (const GLfloat *)&mtx_L2W.a[0]));
            },
            [](const std::shared_ptr<hitable> &target)
            {
                (void)target;
                return true;
            },
            ctxt);

        // Revert state.
        CALL_GL_API(::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE));
    }

    void RasterizeRenderer::setColor(const vec4 &color)
    {
        auto hColor = m_shader.getHandle("color");
        CALL_GL_API(::glUniform4f(hColor, color.x, color.y, color.z, 1.0f));
    }
}
