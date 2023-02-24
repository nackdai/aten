#pragma once

#include <array>
#include <tuple>

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"

#include "kernel/pt_params.h"
#include "kernel/pt_standard_impl.h"
#include "sampler/sampler.h"

#include "reservior.h"
#include "restir_info.h"

namespace idaten
{
    class ReSTIRPathTracing : public StandardPT {
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
            const TileDomain& tileDomain,
            int32_t maxSamples,
            int32_t maxBounce) override;

        virtual void update(
            GLuint gltex,
            int32_t width, int32_t height,
            const aten::CameraParameter& camera,
            const std::vector<aten::GeometryParameter>& shapes,
            const std::vector<aten::MaterialParameter>& mtrls,
            const std::vector<aten::LightParameter>& lights,
            const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
            const std::vector<aten::TriangleParameter>& prims,
            uint32_t advancePrimNum,
            const std::vector<aten::vertex>& vtxs,
            uint32_t advanceVtxNum,
            const std::vector<aten::mat4>& mtxs,
            const std::vector<TextureResource>& texs,
            const EnvmapResource& envmapRsc) override;

        void updateBVH(
            const std::vector<aten::GeometryParameter>& geoms,
            const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
            const std::vector<aten::mat4>& mtxs);

        void updateGeometry(
            std::vector<CudaGLBuffer>& vertices,
            uint32_t vtxOffsetCount,
            TypedCudaMemory<aten::TriangleParameter>& triangles,
            uint32_t triOffsetCount);

        void setGBuffer(
            GLuint gltexGbuffer,
            GLuint gltexMotionDepthbuffer);

        Mode getMode() const
        {
            return m_mode;
        }
        void setMode(Mode mode)
        {
            auto prev = m_mode;
            m_mode = mode;
            if (prev != m_mode) {
                reset();
            }
        }

        ReSTIRMode getReSTIRMode() const
        {
            return m_restirMode;
        }
        void setReSTIRMode(ReSTIRMode mode)
        {
            const auto prev = m_restirMode;
            m_restirMode = mode;
            if (prev != m_restirMode) {
                reset();
            }
        }

        AOVMode getAOVMode() const
        {
            return m_aovMode;
        }
        void setAOVMode(AOVMode mode)
        {
            m_aovMode = mode;
        }

        virtual void reset() override final
        {
            m_frame = 1;
        }

        uint32_t frame() const
        {
            return m_frame;
        }

        void willPickPixel(int32_t ix, int32_t iy)
        {
            m_willPicklPixel = true;
            m_pickedInfo.ix = ix;
            m_pickedInfo.iy = iy;
        }

        bool getPickedPixelInfo(PickedInfo& ret)
        {
            bool isValid = (m_pickedInfo.ix >= 0);

            ret = m_pickedInfo;

            m_pickedInfo.ix = -1;
            m_pickedInfo.iy = -1;

            return isValid;
        }

        void setHitDistanceLimit(float d)
        {
            m_hitDistLimit = d;
        }

        bool canSSRTHitTest() const
        {
            return m_canSSRTHitTest;
        }

        void setCanSSRTHitTest(bool f)
        {
            m_canSSRTHitTest = f;
        }

        bool isEnableProgressive() const
        {
            return m_enableProgressive;
        }
        void setEnableProgressive(bool b)
        {
            m_enableProgressive = b;
        }

    protected:
        virtual bool initPath(
            int32_t width, int32_t height) override final
        {
            if (StandardPT::initPath(width, height)) {
                m_restir_infos.init(width * height);

                for (auto& r : m_reservoirs) {
                    r.init(width * height);
                }

                return true;
            }

            return false;
        }

        void initReSTIR(int32_t width, int32_t height);

        void onRender(
            const TileDomain& tileDomain,
            int32_t width, int32_t height,
            int32_t maxSamples,
            int32_t maxBounce,
            cudaSurfaceObject_t outputSurf,
            cudaTextureObject_t vtxTexPos,
            cudaTextureObject_t vtxTexNml);

        virtual void onHitTest(
            int32_t width, int32_t height,
            int32_t bounce,
            cudaTextureObject_t texVtxPos);

        void missShade(
            int32_t width, int32_t height,
            int32_t bounce,
            int32_t offsetX = -1,
            int32_t offsetY = -1)
        {
            StandardPT::missShade(
                width, height,
                bounce,
                aov_.normal_depth(),
                aov_.albedo_meshid(),
                offsetX, offsetY);
        }

        virtual void onShade(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height,
            int32_t sample,
            int32_t bounce, int32_t rrBounce,
            cudaTextureObject_t texVtxPos,
            cudaTextureObject_t texVtxNml);

        void onShadeReSTIR(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height,
            int32_t sample,
            int32_t bounce, int32_t rrBounce,
            cudaTextureObject_t texVtxPos,
            cudaTextureObject_t texVtxNml);

        void onShadeByShadowRay(
            int32_t bounce,
            cudaTextureObject_t texVtxPos);

        void onShadeByShadowRayReSTIR(
            int32_t width, int32_t height,
            int32_t bounce,
            cudaTextureObject_t texVtxPos,
            cudaTextureObject_t texVtxNml);

        int32_t computelReuse(
            int32_t width, int32_t height,
            int32_t bounce,
            cudaTextureObject_t texVtxPos,
            cudaTextureObject_t texVtxNml);

        virtual void onGather(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height,
            int32_t maxSamples);

        void onDisplayAOV(
            cudaSurfaceObject_t outputSurf,
            int32_t width, int32_t height,
            cudaTextureObject_t texVtxPos);

        void pick(
            int32_t ix, int32_t iy,
            int32_t width, int32_t height,
            cudaTextureObject_t texVtxPos);

        bool isFirstFrame() const
        {
            return (m_frame == 1);
        }

        void setStream(cudaStream_t stream);

    protected:
        idaten::TypedCudaMemory<ReSTIRInfo> m_restir_infos;

        // NOTE
        // previous と spatial destination は使いまわすので２で足りる.
        // 最初の temporal reuse で previous を参照したら
        // 後段の spatila reuse では不要なので、spatial destination にすることができる.
        // e.g.
        //  - frame 1
        //     cur:0
        //     prev:N/A (最初なので temporal は skip)
        //     spatial_dst:1
        //     pos=0 -> pos=1(for next)
        //  - frame 2
        //     cur:1(=pos)
        //     prev:0
        //     spatial_dst:0 (prev:0 は参照済みなので、新しいもので埋めてもいい)
        //     pos=1 -> pos=0(for next)
        std::array<idaten::TypedCudaMemory<Reservoir>, 2> m_reservoirs;
        int32_t m_curReservoirPos = 0;

        // AOV buffer
        using AOVHostBuffer = AT_NAME::AOVHostBuffer<idaten::TypedCudaMemory<float4>, AT_NAME::AOVBufferType::NumBasicAovBuffer>;
        AOVHostBuffer aov_;

        aten::mat4 m_mtxW2V;    // World - View.
        aten::mat4 m_mtxV2C;    // View - Clip.
        aten::mat4 m_mtxC2V;    // Clip - View.

        // View - World.
        aten::mat4 m_mtxV2W;
        aten::mat4 m_mtxPrevW2V;

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

        bool m_enableProgressive{ false };
    };
}
