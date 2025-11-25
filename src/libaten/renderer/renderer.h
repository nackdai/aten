#pragma once

#include "types.h"
#include "math/vec3.h"
#include "math/vec4.h"
#include "renderer/background.h"
#include "renderer/film.h"
#include "scene/host_scene_context.h"
#include "scene/scene.h"
#include "camera/camera.h"

namespace aten
{
    class FBO;

    struct Destination {
        int32_t width{ 0 };
        int32_t height{ 0 };
        uint32_t maxDepth{ 1 };
        uint32_t russianRouletteDepth{ 1 };
        uint32_t sample{ 1 };
        Film* buffer{ nullptr };
        Film* variance{ nullptr };
    };

    class Renderer {
    protected:
        Renderer() = default;
        virtual ~Renderer() = default;

    public:
        void render(
            const context& ctxt,
            Destination& dst,
            scene* scene,
            Camera* camera)
        {
            OnRender(ctxt, dst, scene, camera);
            frame_count_++;
        }

        void SetBG(const BackgroundResource& bg)
        {
            bg_ = bg;
        }

        inline uint32_t GetFrameCount() const noexcept
        {
            return frame_count_;
        }

        void reset() noexcept
        {
            frame_count_ = 1;
        }

        void SetEnableEnvmap(bool b)
        {
            enable_envmap_ = b;
        }

    protected:
        virtual void OnRender(
            const context& ctxt,
            Destination& dst,
            scene* scene,
            Camera* camera) = 0;

        static inline bool isInvalidColor(const vec3& v)
        {
            bool b = isInvalid(v);
            if (!b) {
                if (v.x < 0 || v.y < 0 || v.z < 0) {
                    b = true;
                }
            }

            return b;
        }

    protected:
        BackgroundResource bg_;
        uint32_t frame_count_{ 0 };
        bool enable_envmap_{ true };
    };
}
