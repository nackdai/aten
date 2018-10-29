#pragma once

#include "visualizer/MultiPassPostProc.h"
#include "texture/texture.h"
#include "scene/context.h"

namespace aten {
    class ATrousDenoiser : public MultiPassPostProc {
    public:
        ATrousDenoiser() {}
        ~ATrousDenoiser() {}

    public:
        bool init(
            context& ctxt,
            int width, int height,
            const char* vsPath, const char* fsPath,
            const char* finalVsPath, const char* finalFsPath);

        texture* getNormalMap()
        {
            return m_normal;
        }
        texture* getPositionMap()
        {
            return m_pos;
        }
        texture* getAlbedoMap()
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
            int m_idx{ -1 };
        };

        class ATrousFinalPass : public ATrousPass {
            virtual void prepareRender(
                const void* pixels,
                bool revert) override;
        };

        static const int ITER = 5;

        texture* m_pos{ nullptr };
        texture* m_normal{ nullptr };
        texture* m_albedo{ nullptr };

        ATrousPass m_pass[ITER];
        ATrousFinalPass m_final;
    };
}