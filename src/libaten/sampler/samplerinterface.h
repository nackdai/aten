#pragma once

#include <vector>
#include "defs.h"
#include "types.h"
#include "math/vec2.h"

#ifndef __AT_CUDA__
#define AT_VIRTUAL(f)                   virtual f
#define AT_VIRTUAL_OVERRIDE_FINAL(f)    virtual f override final
#define AT_INHERIT(c)    : public c
#endif

namespace aten {
#ifndef __AT_CUDA__
    class sampler {
    public:
        sampler() {}
        virtual ~sampler() {}

        virtual void init(uint32_t seed, const void* data = nullptr)
        {
            // Nothing is done...
        }
        virtual AT_DEVICE_API real nextSample() = 0;

        virtual AT_DEVICE_API vec2 nextSample2D()
        {
            vec2 ret;
            ret.x = nextSample();
            ret.y = nextSample();

            return ret;
        }
    };
#endif

    void initSampler(
        uint32_t width, uint32_t height,
        int seed = 0,
        bool needInitHalton = false);

    const std::vector<uint32_t>& getRandom();
    uint32_t getRandom(uint32_t idx);

    inline real drand48()
    {
        return (real)::rand() / RAND_MAX;
    }
}
