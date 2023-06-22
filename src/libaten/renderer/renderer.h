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
    struct Destination {
        int32_t width{ 0 };
        int32_t height{ 0 };
        uint32_t maxDepth{ 1 };
        uint32_t russianRouletteDepth{ 1 };
        uint32_t startDepth{ 0 };
        uint32_t sample{ 1 };
        uint32_t mutation{ 1 };
        uint32_t mltNum{ 1 };
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
            camera* camera)
        {
            context::pinContext(&ctxt);
            onRender(ctxt, dst, scene, camera);
            context::removePinnedContext();
        }

        void setBG(background* bg)
        {
            m_bg = bg;
        }

        virtual void enableFeatureLine(bool e) {
            (void)e;
        }

    protected:
        virtual void onRender(
            const context& ctxt,
            Destination& dst,
            scene* scene,
            camera* camera) = 0;

        virtual vec3 sampleBG(const ray& inRay) const
        {
            if (m_bg) {
                return m_bg->sample(inRay);
            }
            return vec3();
        }

        bool hasBG() const
        {
            return (m_bg != nullptr);
        }

        background* bg()
        {
            return m_bg;
        }

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

    private:
        background* m_bg{ nullptr };
    };
}
