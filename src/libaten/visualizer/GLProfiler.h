#pragma once

#include "defs.h"
#include "types.h"

namespace aten {
    class GLProfiler {
    private:
        GLProfiler() = default;
        ~GLProfiler() = default;

        GLProfiler(const GLProfiler&) = delete;
        GLProfiler& operator=(const GLProfiler&) = delete;

    public:
        static void start();
        static void terminate();

        static void begin();
        static double end();

        static void enable();
        static void disable();

        static void trigger();

        static bool isEnabled();

    private:
        static uint32_t s_query[2];
        static bool s_isProfiling;

        static bool s_enable;
    };
}