#pragma once

#include "visualizer/pixelformat.h"
#include "visualizer/MultiPassPostProc.h"

namespace aten {
    class BloomEffect : public MultiPassPostProc {
    public:
        BloomEffect() {}
        virtual ~BloomEffect() {}

    public:
        bool init(
            int width, int height,
            PixelFormat inFmt, PixelFormat outFmt,
            const char* pathVS,
            const char* pathFS_4x4,
            const char* pathFS_2x2,
            const char* pathFS_HBlur,
            const char* pathFS_VBlur,
            const char* pathFS_GaussBlur,
            const char* pathFS_Final);

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
            return m_passFinal.getFbo().getWidth();
        }
        virtual uint32_t getOutHeight() const override final
        {
            return m_passFinal.getFbo().getHeight();
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
                int srcWidth, int srcHeight,
                int dstWidth, int dstHeight,
                PixelFormat inFmt, PixelFormat outFmt,
                const char* pathVS,
                const char* pathFS);

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

            int m_srcWidth;
            int m_srcHeight;

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
