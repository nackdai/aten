#pragma once

#include "camera/camera.h"
#include "math/mat4.h"
#include "renderer/aov.h"


namespace AT_NAME
{
    struct SVGFAovMode : public AT_NAME::AOVType {
        enum Type {
            ObjId = AT_NAME::AOVType::MeshId,
            TexColor = AT_NAME::AOVType::Albedo,
            WireFrame = AT_NAME::AOVType::BeginOfInheritType,
            BaryCentric,
            Motion,
            end_of_AOVMode = Motion,
        };

        static constexpr size_t Num = static_cast<size_t>(Type::end_of_AOVMode) + 1;

        AT_DEVICE_API SVGFAovMode() : AT_NAME::AOVType(AT_NAME::AOVType::Normal) {}
        AT_DEVICE_API ~SVGFAovMode() = default;
        AT_DEVICE_API SVGFAovMode(int32_t type) : AT_NAME::AOVType(static_cast<Type>(type)) {}
    };

    struct SVGFAovBufferType : public AT_NAME::AOVBufferType {
        enum Type {
            ColorVariance = AT_NAME::AOVBufferType::BeginOfInheritType,
            MomentTemporalWeight,
            end_of_AOVBuffer = MomentTemporalWeight,
        };

        static constexpr size_t Num = static_cast<size_t>(Type::end_of_AOVBuffer) + 1;

        AT_DEVICE_API SVGFAovBufferType() = default;
        AT_DEVICE_API ~SVGFAovBufferType() = default;
        AT_DEVICE_API SVGFAovBufferType(int32_t type) : AT_NAME::AOVBufferType(static_cast<Type>(type)) {}
    };

    struct SVGFMtxPack {
        aten::mat4 mtx_W2V;        // World - View.
        aten::mat4 mtx_V2C;        // View - Clip.
        aten::mat4 mtx_C2V;        // Clip - View.

        // View - World.
        aten::mat4 mtx_V2W;
        aten::mat4 mtx_PrevW2V;

        aten::mat4 GetW2C() const
        {
            return mtx_W2V * mtx_V2C;
        }

        void Reset(const aten::CameraParameter& camera)
        {
            mtx_PrevW2V = mtx_W2V;

            camera::ComputeCameraMatrices(camera, mtx_W2V, mtx_V2C);
            mtx_C2V = mtx_V2C * mtx_W2V;
            mtx_V2W = mtx_W2V.invert();
        }
    };

    template <typename BufferContainer>
    struct SVGFParams {
        using buffer_container_type = BufferContainer;
        using buffer_value_type = typename buffer_container_type::value_type;

        // Current AOV buffer position.
        int32_t cur_aov_pos{ 0 };

        using AOVHostBuffer = AT_NAME::AOVHostBuffer<BufferContainer, SVGFAovBufferType::Num>;
        std::array<AOVHostBuffer, 2> aovs;  // AOV buffer. Current frame and previous frame.

        AOVHostBuffer& GetCurAovBuffer()
        {
            return aovs[cur_aov_pos];
        }

        void UpdateCurAovBufferPos()
        {
            cur_aov_pos = 1 - cur_aov_pos;
        }

        void InitBuffers(int32_t width, int32_t height)
        {
            for (auto& aov : aovs) {
                aov.traverse(
                    [width, height](BufferContainer& buffer)
                    {
                        if (buffer.empty()) {
                            buffer.resize(width * height);
                        }
                    });
            }

            temporary_color_buffer.resize(width * height);
        }

        // For A-trous wavelet.
        std::array<BufferContainer, 2> atrous_clr_variance;

        BufferContainer temporary_color_buffer;

        int32_t atrous_iter_cnt{ 5 };

        SVGFMtxPack mtxs;
    };
}
