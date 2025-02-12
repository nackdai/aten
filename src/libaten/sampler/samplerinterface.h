#pragma once

#include <vector>
#include "defs.h"
#include "types.h"
#include "math/vec2.h"

#ifndef __AT_CUDA__
#define AT_VIRTUAL(f)                   virtual f
#define AT_VIRTUAL_OVERRIDE(f)          virtual f override
#define AT_VIRTUAL_OVERRIDE_FINAL(f)    virtual f override final
#define AT_INHERIT(c)    : public c
#endif

namespace aten {
#ifndef __AT_CUDA__
    class sampler_interface {
    public:
        sampler_interface() {}
        virtual ~sampler_interface() {}

        virtual void init(uint32_t seed, const void* data = nullptr)
        {
            // Nothing is done...
        }
        virtual AT_HOST_DEVICE_API float nextSample() = 0;

        virtual AT_HOST_DEVICE_API vec2 nextSample2D()
        {
            vec2 ret;
            ret.x = nextSample();
            ret.y = nextSample();

            return ret;
        }
    };
#endif

    void initSampler(
        int32_t width, int32_t height,
        int32_t seed = 0);

    const std::vector<uint32_t>& getRandom();
    uint32_t getRandom(uint32_t idx);

    inline float drand48()
    {
        return static_cast<float>(::rand() / RAND_MAX);
    }
}
