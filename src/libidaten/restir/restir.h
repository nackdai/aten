#pragma once

#include <array>
#include <tuple>

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"

#include "kernel/pathtracing.h"
#include "renderer/pathtracing/pt_params.h"
#include "renderer/restir/restir_types.h"
#include "sampler/sampler.h"

namespace idaten
{
    class ReSTIRPathTracing : public PathTracingImplBase {
    public:
        enum class Mode {
            ReSTIR,
            PT,
            AOVar,  // Arbitrary Output Variables.
        };

        enum class ReSTIRMode {
            ReSTIR,
            SpatialReuse,
            TemporalReuse,
        };

        enum AOVMode {
            Normal,
            TexColor,
            Depth,
            WireFrame,
            BaryCentric,
            Motion,
            ObjId,
        };

        struct NormalMaterialStorage {
            aten::vec3 normal;
            struct {
                uint32_t is_voxel : 1;
                uint32_t is_mtrl_valid : 1;
                uint32_t mtrl_idx : 16;
            };
        };

        struct PickedInfo {
            int32_t ix{ -1 };
            int32_t iy{ -1 };
            aten::vec3 color;
            aten::vec3 normal;
            float depth;
            int32_t meshid;
            int32_t triid;
            int32_t mtrlid;
        };

        ReSTIRPathTracing() = default;
        virtual ~ReSTIRPathTracing() = default;

        virtual void render(
            int32_t width, int32_t height,
            int32_t maxSamples,
            int32_t maxBounce) override;

        void UpdateSceneData(
            GLuint gltex,
            int32_t width, int32_t height,
            const aten::CameraParameter& camera,
            const aten::context& scene_ctxt,
            const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
            uint32_t advance_prim_num,
            uint32_t advance_vtx_num,
            const aten::BackgroundResource& bg_resource,
            std::function<const aten::Grid* (const aten::context&)> proxy_get_grid_from_host_scene_context = nullptr) override;

        void SetGBuffer(
            GLuint gltexGbuffer,
            GLuint gltexMotionDepthbuffer);

        Mode GetMode() const
        {
            return m_mode;
        }
        void SetMode(Mode mode)
        {
            auto prev = m_mode;
            m_mode = mode;
            if (prev != m_mode) {
                reset();
            }
        }

        ReSTIRMode GetReSTIRMode() const
        {
            return m_restirMode;
        }
        void SetReSTIRMode(ReSTIRMode mode)
        {
            const auto prev = m_restirMode;
            m_restirMode = mode;
            if (prev != m_restirMode) {
                reset();
            }
        }

        AOVMode GetAOVMode() const
        {
            return m_aovMode;
        }
        void SetAOVMode(AOVMode mode)
        {
            m_aovMode = mode;
        }

        void WillPickPixel(int32_t ix, int32_t iy)
        {
            m_willPicklPixel = true;
            m_pickedInfo.ix = ix;
            m_pickedInfo.iy = iy;
        }

        bool GetPickedPixelInfo(PickedInfo& ret)
        {
            bool isValid = (m_pickedInfo.ix >= 0);

            ret = m_pickedInfo;

            m_pickedInfo.ix = -1;
            m_pickedInfo.iy = -1;

            return isValid;
        }

        bool CanSSRTHitTest() const
        {
            return m_canSSRTHitTest;
        }

        void SetCanSSRTHitTest(bool f)
        {
            m_canSSRTHitTest = f;
        }

        bool IsEnableProgressive() const
        {
            return m_enableProgressive;
        }
        void SetEnableProgressive(bool b)
        {
            m_enableProgressive = b;
        }

    protected:
        virtual bool InitPath(
            int32_t width, int32_t height) override final
        {
            if (StandardPT::InitPath(width, height)) {
                size_t size = width * height;
                m_restir_infos.Init(size);
                m_reservoirs.Init(size);
                return true;
            }

            return false;
        }

        void InitReSTIR(int32_t width, int32_t height);

        void OnRender(
            int32_t width, int32_t height,
            int32_t maxSamples,
            int32_t maxBounce,
            cudaSurfaceObject_t outputSurf);

        void OnShadeReSTIR(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height,
            int32_t sample,
            int32_t bounce, int32_t rrBounce);

        void OnShadeByShadowRayReSTIR(
            int32_t width, int32_t height,
            int32_t bounce);

        aten::tuple<int32_t, int32_t> ComputelReuse(
            int32_t width, int32_t height,
            int32_t bounce);

        void OnDisplayAOV(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height);

        void pick(
            int32_t ix, int32_t iy,
            int32_t width, int32_t height);

        bool IsFirstFrame() const
        {
            return (m_frame == 1);
        }

        void SetStream(cudaStream_t stream);

    protected:
        AT_NAME::ReuseParams<idaten::TypedCudaMemory<AT_NAME::ReSTIRInfo>> m_restir_infos;

        AT_NAME::ReuseParams<idaten::TypedCudaMemory<AT_NAME::Reservoir>> m_reservoirs;

        AT_NAME::MatricesForRendering mtxs_;

        // G-Buffer rendered by OpenGL.
        idaten::CudaGLSurface m_gbuffer;
        idaten::CudaGLSurface m_motionDepthBuffer;

        idaten::TypedCudaMemory<PickedInfo> m_pick;

        bool m_willPicklPixel{ false };
        PickedInfo m_pickedInfo;

        Mode m_mode{ Mode::ReSTIR };
        ReSTIRMode m_restirMode{ ReSTIRMode::ReSTIR };
        AOVMode m_aovMode{ AOVMode::WireFrame };

        bool m_isListedTextureObject{ false };
        bool m_canSSRTHitTest{ true };
    };
}
