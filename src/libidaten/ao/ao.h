#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"
#include "kernel/pathtracing.h"

namespace idaten
{
    class AORenderer : public PathTracing {
    public:
        AORenderer() = default;
        ~AORenderer() = default;

    public:
        int32_t getNumRays() const
        {
            return m_ao_num_rays;
        }
        void setNumRays(int32_t num)
        {
            m_ao_num_rays = num;
        }

        float getRadius() const
        {
            return m_ao_radius;
        }
        void setRadius(float radius)
        {
            m_ao_radius = radius;
        }

    protected:
        void OnRender(
            int32_t width, int32_t height,
            int32_t maxSamples,
            int32_t maxBounce,
            cudaSurfaceObject_t outputSurf) override;

        void ShadeMissAO(
            int32_t width, int32_t height,
            int32_t bounce);

        void ShadeAO(
            int32_t width, int32_t height,
            int32_t bounce, int32_t rrBounce);

    protected:
        int32_t m_ao_num_rays{ 1 };
        float m_ao_radius{ 1.0f };
    };
}
