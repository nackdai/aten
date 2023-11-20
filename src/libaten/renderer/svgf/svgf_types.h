#pragma once

#include "camera/camera.h"
#include "math/mat4.h"
#include "renderer/aov.h"


namespace AT_NAME
{
    /**
     * @brief Enum for SVGF AOV display modes inherit from the normal AOV display modes.
     */
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

    /**
     * @brief Enum for SVGF AOV buffer types inherit from the normal AOV buffer types.
     */
    struct SVGFAovBufferType : public AT_NAME::AOVBufferType {
        enum Type {
            ColorVariance = AT_NAME::AOVBufferType::BeginOfInheritType, ///< Color and variance.
            MomentTemporalWeight,                                       ///< Moments and temporal weight.
            end_of_AOVBuffer = MomentTemporalWeight,                    ///< End of type.
        };

        static constexpr size_t Num = static_cast<size_t>(Type::end_of_AOVBuffer) + 1;

        AT_DEVICE_API SVGFAovBufferType() = default;
        AT_DEVICE_API ~SVGFAovBufferType() = default;
        AT_DEVICE_API SVGFAovBufferType(int32_t type) : AT_NAME::AOVBufferType(static_cast<Type>(type)) {}
    };

    /**
     * @brief Matrices for SVGF.
     */
    struct SVGFMtxPack {
        aten::mat4 mtx_W2V;         ///< Matrix to convert from World coordinate to View cooridnate.
        aten::mat4 mtx_V2C;         ///< Matrix to convert from View coordinate to Clip cooridnate.
        aten::mat4 mtx_C2V;         ///< Matrix to convert from Clip coordinate to View cooridnate.

        aten::mat4 mtx_V2W;         ///< Matrix to convert from View coordinate to World cooridnate.
        aten::mat4 mtx_PrevW2V;     ///< Matrix to convert from World coordinate to View cooridnate in the previous frame.

        /**
         * @param Get a matrix to convert from World coordinate to Clip cooridnate.
         *
         * @return Matrix to convert from World coordinate to Clip cooridnate.
         */
        aten::mat4 GetW2C() const
        {
            return mtx_W2V * mtx_V2C;
        }

        /**
         * @brief Reset the matrices with the specified camera parameter.
         *
         * @param[in] camera Camera parameter to reset the matrices.
         */
        void Reset(const aten::CameraParameter& camera)
        {
            mtx_PrevW2V = mtx_W2V;

            camera::ComputeCameraMatrices(camera, mtx_W2V, mtx_V2C);
            mtx_C2V = mtx_V2C * mtx_W2V;
            mtx_V2W = mtx_W2V.invert();
        }
    };

    /**
     * @brief SVGF parameters to be managed in host.
     *
     * @tparam BufferContainer Buffer container type in host.
     * @tparam BufferContainerForMotionDepth Buffer container type spectialized for motion depth buffer.
     */
    template <typename BufferContainer, typename BufferContainerForMotionDepth>
    struct SVGFParams {
        using buffer_container_type = BufferContainer;
        using buffer_value_type = typename buffer_container_type::value_type;

        int32_t curr_aov_pos{ 0 };  ///< Current AOV buffer position.

        using AOVHostBuffer = AT_NAME::AOVHostBuffer<BufferContainer, SVGFAovBufferType::Num>;
        std::array<AOVHostBuffer, 2> aovs;  ///< Array of AOV buffer to store current frame one and previous frame one.

        /** Buffer to store color and various for ping-pong A-trous wavelet filter interations. */
        std::array<BufferContainer, 2> atrous_clr_variance;

        /** Buffer to store the midstream fitered color temporary. */
        BufferContainer temporary_color_buffer;

        BufferContainerForMotionDepth motion_depth_buffer;  ///< Buffer to store motion and depth.

        int32_t atrous_iter_cnt{ 5 };   ///< Count of A-trous wavelet filter iterations.

        SVGFMtxPack mtxs;   ///< Matrices for SVGF.

        /**
         * @brief Get AOV buffer of the current frame
         *
         * @return AOV buffer of the current frame
         */
        AOVHostBuffer& GetCurrAovBuffer()
        {
            return aovs[curr_aov_pos];
        }

        /**
         * @brief Get AOV buffer of the previous frame
         *
         * @return AOV buffer of the previous frame
         */
        AOVHostBuffer& GetPrevAovBuffer()
        {
            const auto prev_aov_pos = GetPrevAocPos();
            return aovs[prev_aov_pos];
        }

        /**
         * @brief Update current AOV buffer position.
         */
        void UpdateCurrAovBufferPos()
        {
            curr_aov_pos = 1 - curr_aov_pos;
        }

        /**
         * @brief Get current AOV buffer position.
         *
         * @return Current AOV buffer position.
         */
        int32_t GetCurrAovPos() const
        {
            return curr_aov_pos;
        }

        /**
         * @brief Get previous AOV buffer position.
         *
         * @return Previous AOV buffer position.
         */
        int32_t GetPrevAocPos() const
        {
            return 1 - curr_aov_pos;
        }

        /**
         * @brief Initialize buffers.
         *
         * @param[in] width Screen width.
         * @param[in] height Screen height.
         */
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
    };
}
