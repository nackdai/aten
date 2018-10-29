#include <algorithm>
#include <iterator>

#include "deformable/deformable.h"
#include "deformable/DeformAnimation.h"
#include "misc/stream.h"
#include "visualizer/shader.h"
#include "visualizer/atengl.h"
#include "texture/texture.h"
#include "camera/camera.h"
#include "accelerator/accelerator.h"

namespace aten
{
    deformable::~deformable()
    {
        if (m_accel) {
            delete m_accel;
        }
    }

    bool deformable::read(const char* path)
    {
        FileInputStream file;
        AT_VRETURN_FALSE(file.open(path, "rb"));

        FileInputStream* stream = &file;

        MdlHeader header;
        AT_VRETURN_FALSE(AT_STREAM_READ(stream, &header, sizeof(header)));

        // Mesh.
        {
            MdlChunkHeader meshChunkHeader;
            AT_VRETURN_FALSE(AT_STREAM_READ(stream, &meshChunkHeader, sizeof(meshChunkHeader)));

            if (meshChunkHeader.magicChunk == MdlChunkMagic::Mesh) {
                AT_VRETURN_FALSE(m_mesh.read(stream));
            }
            else {
                AT_VRETURN_FALSE(false);
            }
        }

        auto pos = stream->curPos();

        // Skeleton.
        {
            MdlChunkHeader sklChunkHeader;
            AT_VRETURN_FALSE(AT_STREAM_READ(stream, &sklChunkHeader, sizeof(sklChunkHeader)));

            if (sklChunkHeader.magicChunk == MdlChunkMagic::Joint) {
                AT_VRETURN_FALSE(m_skl.read(stream));
            }
            else {
                AT_VRETURN_FALSE(false);
            }

            m_sklController.init(&m_skl);
        }

        return true;
    }

    void deformable::release()
    {
        m_mesh.release();
        m_skl.release();
        m_sklController.release();
    }

    class DeformMeshRenderHelper : public IDeformMeshRenderHelper {
    public:
        DeformMeshRenderHelper(shader* s) : m_shd(s) {}
        virtual ~DeformMeshRenderHelper() {}

        virtual void applyMatrix(uint32_t idx, const mat4& mtx) override final
        {
            if (m_handleMtxJoint < 0) {
                m_handleMtxJoint = m_shd->getHandle("mtxJoint");
                m_mtxs.reserve(4);
            }

            m_mtxs.push_back(mtx);
        }

        virtual void applyMaterial(const context& ctxt, const MeshMaterial& mtrlDesc) override final
        {
            const auto mtrl = ctxt.findMaterialByName(mtrlDesc.name);

            if (mtrl) {
                const auto& mtrlParam = mtrl->param();

                auto albedo = const_cast<texture*>(ctxt.getTexture(mtrlParam.albedoMap));
                if (albedo) {
                    albedo->initAsGLTexture();
                    albedo->bindAsGLTexture(0, m_shd);
                }
            }
        }

        virtual void commitChanges(bool isGPUSkinning, uint32_t triOffset) override final
        {
            if (isGPUSkinning) {
                return;
            }

            AT_ASSERT(m_handleMtxJoint >= 0);

            uint32_t mtxNum = (uint32_t)m_mtxs.size();

            CALL_GL_API(::glUniformMatrix4fv(m_handleMtxJoint, mtxNum, GL_TRUE, (const GLfloat*)&m_mtxs[0]));

            m_mtxs.clear();
        }

        shader* m_shd{ nullptr };
        int m_handleMtxJoint{ -1 };
        std::vector<mat4> m_mtxs;
    };

    void deformable::render(
        const context& ctxt,
        shader* shd)
    {
        AT_ASSERT(shd);

        DeformMeshRenderHelper helper(shd);

        m_mesh.render(ctxt, m_sklController, &helper);
    }

    void deformable::update(const mat4& mtxL2W)
    {
        m_sklController.buildPose(mtxL2W);
    }

    void deformable::update(
        const mat4& mtxL2W,
        real time,
        DeformAnimation* anm)
    {
        if (anm) {
            anm->applyAnimation(&m_sklController, time);
        }
        m_sklController.buildPose(mtxL2W);
    }

    void deformable::getGeometryData(
        const context& ctxt,
        std::vector<SkinningVertex>& vtx,
        std::vector<uint32_t>& idx,
        std::vector<aten::PrimitiveParamter>& tris) const
    {
        m_mesh.getGeometryData(ctxt, vtx, idx, tris);
    }

    const std::vector<mat4>& deformable::getMatrices() const
    {
        return m_sklController.getMatrices();
    }

    void deformable::build()
    {
        if (!m_accel) {
            m_accel = accelerator::createAccelerator(AccelType::UserDefs);
        }

        const auto& desc = m_mesh.getDesc();

        // TODO
        // Not applied animation...
        setBoundingBox(aabb(
            aten::vec3(desc.minVtx[0], desc.minVtx[1], desc.minVtx[2]),
            aten::vec3(desc.maxVtx[0], desc.maxVtx[1], desc.maxVtx[2])));
    }

    class DeformMeshRenderHelperEx : public IDeformMeshRenderHelper {
    public:
        DeformMeshRenderHelperEx() {}
        virtual ~DeformMeshRenderHelperEx() {}

        virtual void applyMatrix(uint32_t idx, const mat4& mtx) override final {}
        virtual void applyMaterial(const context& ctxt, const MeshMaterial& mtrlDesc) override final {}
        virtual void commitChanges(bool isGPUSkinning, uint32_t triOffset) override final
        {
            AT_ASSERT(isGPUSkinning)
            func(mtxL2W, mtxPrevL2W, objid, triOffset + globalTriOffset);
        }

        aten::hitable::FuncPreDraw func;
        aten::mat4 mtxL2W;
        aten::mat4 mtxPrevL2W;
        int objid;
        uint32_t globalTriOffset;
    };

    void deformable::drawForGBuffer(
        aten::hitable::FuncPreDraw func,
        const context& ctxt,
        const aten::mat4& mtxL2W,
        const aten::mat4& mtxPrevL2W,
        int parentId,
        uint32_t triOffset)
    {
        int objid = (parentId < 0 ? id() : parentId);

        DeformMeshRenderHelperEx helper;
        {
            helper.func = func;
            helper.mtxL2W = mtxL2W;
            helper.mtxPrevL2W = mtxPrevL2W;
            helper.objid = objid;
            helper.globalTriOffset = triOffset;
        }

        m_mesh.render(ctxt, m_sklController, &helper);
    }

    void deformable::initGLResources(shader* shd)
    {
        if (m_isInitializedToRender) {
            return;
        }

        m_mesh.initGLResources(shd);
        m_isInitializedToRender = true;
    }

    void deformable::initGLResourcesWithDeformableRenderer(DeformableRenderer& renderer)
    {
        auto shd = renderer.getShader();
        initGLResources(shd);
    }

    //////////////////////////////////////////////////////////////

    bool DeformableRenderer::init(
        int width, int height,
        const char* pathVS,
        const char* pathFS)
    {
        return m_shd.init(width, height, pathVS, pathFS);
    }

    void DeformableRenderer::render(
        const context& ctxt,
        const camera* cam,
        deformable* mdl)
    {
        CALL_GL_API(::glClearColor(0, 0.5f, 1.0f, 1.0f));
        CALL_GL_API(::glClearDepthf(1.0f));
        CALL_GL_API(::glClearStencil(0));
        CALL_GL_API(::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

        CALL_GL_API(::glEnable(GL_DEPTH_TEST));

        // For Alpha Blend.
        CALL_GL_API(::glEnable(GL_BLEND));
        CALL_GL_API(::glBlendEquation(GL_FUNC_ADD));
        CALL_GL_API(::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

        m_shd.prepareRender(nullptr, false);

        {
            auto camparam = cam->param();

            // TODO
            camparam.znear = real(0.1);
            camparam.zfar = real(10000.0);

            mat4 mtxW2V;
            mat4 mtxV2C;

            mtxW2V.lookat(
                camparam.origin,
                camparam.center,
                camparam.up);

            mtxV2C.perspective(
                camparam.znear,
                camparam.zfar,
                camparam.vfov,
                camparam.aspect);

            aten::mat4 mtxW2C = mtxV2C * mtxW2V;

            auto hMtxW2C = m_shd.getHandle("mtxW2C");
            CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, &mtxW2C.a[0]));

            // NOTE
            // グローバルマトリクス計算時にルートに local to world マトリクスは乗算済み.
            // そのため、シェーダでは計算する必要がないので、シェーダに渡さない.
        }

        mdl->render(ctxt, &m_shd);
    }
}
