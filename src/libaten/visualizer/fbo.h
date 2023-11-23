#pragma once

#include <vector>
#include <functional>

#include "types.h"
#include "misc/span.h"
#include "visualizer/pixelformat.h"

namespace aten {
    /**
     * @brief Frame Buffer Object.
     */
    class FBO {
    public:
        FBO() = default;
        ~FBO() = default;

        FBO(const FBO&) = delete;
        FBO(FBO&&) = delete;
        FBO& operator=(const FBO&) = delete;
        FBO& operator=(FBO&&) = delete;

    public:
        /**
         * @brief Initialize FBO.
         *
         * @param width Width of frame buffer.
         * @param height Height of frame buffer.
         * @param fmt Pixel format of frame buffer.
         * @param need_depth If FBO needs to have the depth buffer, true should be sepecified.
         * @return If initializeing is done properly, returns true. Otherwise, returns false.
         */
        bool init(
            int32_t width,
            int32_t height,
            PixelFormat fmt,
            bool need_depth = false);

        /**
         * @brief Get if FBO is valid.
         *
         * @return If FBO is valid, returns true. Otherwise, returns false.
         */
        bool IsValid() const noexcept
        {
            return (fbo_ > 0);
        }

        /**
         * @brief Bind FBO as the texture.
         *
         * @param idx Index to which FBO should be binded.
         */
        void BindAsTexture(uint32_t idx = 0);

        /**
         * @brief Bind FBO for ready to be rendered.
         *
         * @param need_depth If this is true, depth buffer is also binded.
         */
        void BindFBO(bool need_depth = false);

        /**
         * @brief Get width of frame buffer.
         *
         * @return Width of frame buffer.
         */
        int32_t GetWidth() const noexcept
        {
            return width_;
        }

        /**
         * @brief Get height of frame buffer.
         *
         * @return Height of frame buffer.
         */
        int32_t GetHeight() const noexcept
        {
            return height_;
        }

        /**
         * @brief Get OpenGL texture handle which is binded with FBO.
         *
         * @param idx Index to texture handle.
         * @return OpenGL texture handle.
         */
        uint32_t GetGLTextureHandle(uint32_t idx = 0) const
        {
            return texture_handles_[idx];
        }

        /**
         * @brief Get OpenGL frame buffer handle.
         *
         * @return OpenGL frame buffe handle.
         */
        uint32_t GetGLHandle() const
        {
            return fbo_;
        }

        void asMulti(uint32_t num);

        /**
         * @brief Save the content of the frame buffer.
         *
         * @param dst Destination to save the content of the frame buffer.
         * @param target_idx Index of the frame buffer to be saved.
         */
        void SaveToBuffer(void* dst, int32_t target_idx) const;

        template <class ElementType>
        bool SaveToBuffer(aten::span<ElementType>& dst, int32_t target_idx) const
        {
            const auto bpp = GetBytesPerPxiel(pixel_fmt_);
            const auto bytes = width_ * height_ * bpp;
            AT_ASSERT(dst.size_bytes() == bytes);

            if (dst.size_bytes() == bytes) {
                SaveToBuffer(dst.data(), target_idx);
                return true;
            }
            return false;
        }

        using FuncBindFbo = std::function<void(const std::vector<uint32_t>&, std::vector<uint32_t>&)>;

        /**
         * @brief Set an user defined function to bind FBO.
         *
         * @param func User defined function to bind FBO.
         */
        void SetBindFboFunction(FuncBindFbo func)
        {
            func_bind_fbo_ = func;
        }

    protected:
        uint32_t fbo_{ 0 };

        int32_t m_num{ 1 };

        std::vector<uint32_t> texture_handles_;

        std::vector<uint32_t> target_buffer_attachment_list_;

        FuncBindFbo func_bind_fbo_{ nullptr };

        uint32_t depth_buffer_handle_{ 0 };

        PixelFormat pixel_fmt_{ PixelFormat::rgba8 };
        uint32_t width_{ 0 };
        uint32_t height_{ 0 };
    };
}
