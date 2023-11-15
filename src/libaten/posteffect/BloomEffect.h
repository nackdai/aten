#pragma once

#include "visualizer/pixelformat.h"
#include "visualizer/MultiPassPostProc.h"

namespace aten {
    class BloomEffect : public MultiPassPostProc {
    public:
        BloomEffect() = default;
        virtual ~BloomEffect() = default;

    public:
        bool init(
            int32_t width, int32_t height,
            PixelFormat inFmt, PixelFormat outFmt,
            std::string_view pathVS,
            std::string_view pathFS_4x4,
            std::string_view pathFS_2x2,
            std::string_view pathFS_HBlur,
            std::string_view pathFS_VBlur,
            std::string_view pathFS_GaussBlur,
            std::string_view pathFS_Final);

        void setParam(float threshold, float adaptedLum)
        {
            m_threshold = std::max(threshold, 0.0f);
            m_adaptedLum = std::max(adaptedLum, 0.0f);
        }

        virtual PixelFormat inFormat() const override final
        {
            return m_fmtIn;
        }

        virtual PixelFormat outFormat() const override final
        {
            return m_fmtOut;
        }

        virtual uint32_t getOutWidth() const override final
        {
            return m_passFinal.getFbo().GetWidth();
        }
        virtual uint32_t getOutHeight() const override final
        {
            return m_passFinal.getFbo().GetHeight();
        }

        virtual FBO& getFbo() override final
        {
            return m_passFinal.getFbo();
        }

    private:
        class BloomEffectPass : public visualizer::PostProc {
        public:
            BloomEffectPass() = delete;
            BloomEffectPass(BloomEffect* body) : m_body(body) {}
            virtual ~BloomEffectPass() = default;

        public:
            bool init(
                int32_t srcWidth, int32_t srcHeight,
                int32_t dstWidth, int32_t dstHeight,
                PixelFormat inFmt, PixelFormat outFmt,
                std::string_view pathVS,
                std::string_view pathFS);

            virtual void prepareRender(
                const void* pixels,
                bool revert) override;

            virtual PixelFormat inFormat() const override
            {
                return m_fmtIn;
            }
            virtual PixelFormat outFormat() const override
            {
                return m_fmtOut;
            }

            BloomEffect* m_body;

            int32_t m_srcWidth;
            int32_t m_srcHeight;

            PixelFormat m_fmtIn;
            PixelFormat m_fmtOut;
        };

        class BloomEffectFinalPass : public BloomEffectPass {
        public:
            BloomEffectFinalPass(BloomEffect* body) : BloomEffectPass(body) {}
            virtual ~BloomEffectFinalPass() {}

            virtual void prepareRender(
                const void* pixels,
                bool revert) override;
        };

        visualizer* getVisualizer()
        {
            return PostProc::getVisualizer();
        }

        PixelFormat m_fmtIn;
        PixelFormat m_fmtOut;

        BloomEffectPass m_pass4x4{ BloomEffectPass(this) };
        BloomEffectPass m_pass2x2{ BloomEffectPass(this) };
        BloomEffectPass m_passHBlur{ BloomEffectPass(this) };
        BloomEffectPass m_passVBlur{ BloomEffectPass(this) };
        BloomEffectPass m_passGaussBlur_4x4{ BloomEffectPass(this) };
        BloomEffectPass m_passGaussBlur_2x2{ BloomEffectPass(this) };
        BloomEffectFinalPass m_passFinal{ BloomEffectFinalPass(this) };

        float m_threshold{ 0.15f };
        float m_adaptedLum{ 0.2f };
    };
}
