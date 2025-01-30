#pragma once

#include "visualizer/MultiPassPostProc.h"
#include "image/texture.h"
#include "scene/host_scene_context.h"

namespace aten {
    class ATrousDenoiser : public MultiPassPostProc {
    public:
        ATrousDenoiser() = default;
        ~ATrousDenoiser() = default;

    public:
        bool init(
            context& ctxt,
            int32_t width, int32_t height,
            std::string_view vsPath,
            std::string_view fsPath,
            std::string_view finalVsPath,
            std::string_view finalFsPath);

        std::shared_ptr<texture> getNormalMap()
        {
            return m_normal;
        }
        std::shared_ptr<texture> getPositionMap()
        {
            return m_pos;
        }
        std::shared_ptr<texture> getAlbedoMap()
        {
            return m_albedo;
        }

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

    private:
        class ATrousPass : public visualizer::PostProc {
        public:
            ATrousPass() {}
            virtual ~ATrousPass() {}

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

            ATrousDenoiser* m_body{ nullptr };
            int32_t m_idx{ -1 };
        };

        class ATrousFinalPass : public ATrousPass {
            virtual void prepareRender(
                const void* pixels,
                bool revert) override;
        };

        static const int32_t ITER = 5;

        visualizer* getVisualizer()
        {
            return PostProc::getVisualizer();
        }

        std::shared_ptr<texture> m_pos;
        std::shared_ptr<texture> m_normal;
        std::shared_ptr<texture> m_albedo;

        ATrousPass m_pass[ITER];
        ATrousFinalPass m_final;
    };
}
