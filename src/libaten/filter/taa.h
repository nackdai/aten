#pragma once

#include "visualizer/MultiPassPostProc.h"
#include "image/texture.h"
#include "camera/pinhole.h"
#include "math/mat4.h"

namespace aten {
    class TAA : public MultiPassPostProc {
    public:
        TAA() = default;
        virtual ~TAA() = default;

    public:
        bool init(
            int32_t width, int32_t height,
            std::string_view taaVsPath,
            std::string_view taaFsPath,
            std::string_view finalVsPath,
            std::string_view finalFsPath);

        virtual PixelFormat inFormat() const override final
        {
            return PixelFormat::rgba32f;
        }

        virtual PixelFormat outFormat() const override final
        {
            return PixelFormat::rgba32f;
        }

        virtual FBO& getFbo() override final
        {
            return m_final.getFbo();
        }

        void setMotionDepthBufferHandle(uint32_t handle)
        {
            m_motionDepthBuffer = handle;
        }

        void enableTAA(bool e)
        {
            m_enableTAA = e;
        }
        bool isEnableTAA() const
        {
            return m_enableTAA;
        }

        void showTAADiff(bool s)
        {
            m_canShowAADiff = s;
        }
        bool canShowTAADiff() const
        {
            return m_canShowAADiff;
        }

    private:
        void prepareFbo(const std::vector<uint32_t>& tex, std::vector<uint32_t>& comps);

        class TAAPass : public visualizer::PostProc {
        public:
            TAAPass() {}
            virtual ~TAAPass() {}

        public:
            virtual void prepareRender(
                const void* pixels,
                bool revert) override;

            virtual PixelFormat inFormat() const override final
            {
                return PixelFormat::rgba32f;
            }
            virtual PixelFormat outFormat() const override final
            {
                return PixelFormat::rgba32f;
            }

            TAA* m_body{ nullptr };
        };

        class FinalPass : public visualizer::PostProc {
        public:
            FinalPass() {}
            virtual ~FinalPass() {}

        public:
            virtual void prepareRender(
                const void* pixels,
                bool revert) override;

            virtual PixelFormat inFormat() const override final
            {
                return PixelFormat::rgba32f;
            }
            virtual PixelFormat outFormat() const override final
            {
                return PixelFormat::rgba32f;
            }

            TAA* m_body{ nullptr };
        };

        visualizer* getVisualizer()
        {
            return PostProc::getVisualizer();
        }

    private:
        TAAPass m_taa;
        FinalPass m_final;

        uint32_t m_motionDepthBuffer{ 0 };

        int32_t m_idx{ 0 };

        bool m_enableTAA{ true };
        bool m_canShowAADiff{ false };
    };
}
