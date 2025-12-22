#pragma once

#include "math/cuda_host_common_math.h"
#include "renderer/pathtracing/pt_params.h"

#ifdef __AT_CUDA__
#include "cuda/cudadefs.h"
#include "cuda/cudamemory.h"
#include "cuda/helper_math.h"
#endif

#define AT_ENABLE_PATH_TIME_PROFILE

namespace AT_NAME {
    class PathTimeProfiler {
    public:
        AT_DEVICE_API PathTimeProfiler(PathThroughput& throughput)
#ifdef AT_ENABLE_PATH_TIME_PROFILE
            : throughput_(throughput)
        {
#ifdef __CUDACC__
            // NOTE:
            // https://qiita.com/syo0901/items/7ea3b8dfc01fd5cc2cf4
            asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start_));
#endif
        }
#else
            = default;
#endif
        AT_DEVICE_API ~PathTimeProfiler() = default;

        PathTimeProfiler() = delete;
        PathTimeProfiler(const PathTimeProfiler&) = delete;
        PathTimeProfiler(PathTimeProfiler&&) = delete;
        PathTimeProfiler& operator=(const PathTimeProfiler&) = delete;
        PathTimeProfiler& operator=(PathTimeProfiler&&) = delete;

        AT_DEVICE_API void end()
        {
#ifdef AT_ENABLE_PATH_TIME_PROFILE
#ifdef __CUDACC__
            // NOTE:
            // https://qiita.com/syo0901/items/7ea3b8dfc01fd5cc2cf4
            long long int end;
            asm volatile("mov.u64  %0, %globaltimer;" : "=l"(end));
            const auto elapsed = end - start_;

            constexpr float GlobalTimerToMs = 1e-6F;
            throughput_.time += elapsed * GlobalTimerToMs;
#endif
#else
#endif
        }

    private:
        using TimeType = long long int;

        PathThroughput& throughput_;
        TimeType start_{ 0 };
    };

    inline AT_DEVICE_API aten::v3 ComputeTemperature(float t)
    {
        // https://developer.nvidia.com/blog/profiling-dxr-shaders-with-timer-instrumentation/

        const std::array c = {
            make_float3(0.0f / 255.0f,   2.0f / 255.0f,  91.0f / 255.0f),
            make_float3(0.0f / 255.0f, 108.0f / 255.0f, 251.0f / 255.0f),
            make_float3(0.0f / 255.0f, 221.0f / 255.0f, 221.0f / 255.0f),
            make_float3(51.0f / 255.0f, 221.0f / 255.0f,   0.0f / 255.0f),
            make_float3(255.0f / 255.0f, 252.0f / 255.0f,   0.0f / 255.0f),
            make_float3(255.0f / 255.0f, 180.0f / 255.0f,   0.0f / 255.0f),
            make_float3(255.0f / 255.0f, 104.0f / 255.0f,   0.0f / 255.0f),
            make_float3(226.0f / 255.0f,  22.0f / 255.0f,   0.0f / 255.0f),
            make_float3(191.0f / 255.0f,   0.0f / 255.0f,  83.0f / 255.0f),
            make_float3(145.0f / 255.0f,   0.0f / 255.0f,  65.0f / 255.0f)
        };

        const float s = t * 10.0f;

        const int cur = int(s) <= 9 ? int(s) : 9;
        const int prv = cur >= 1 ? cur - 1 : 0;
        const int nxt = cur < 9 ? cur + 1 : 9;

        const float blur = 0.8f;

        const float wc = smoothstep(float(cur) - blur, float(cur) + blur, s) * (1.0f - smoothstep(float(cur + 1) - blur, float(cur + 1) + blur, s));
        const float wp = 1.0f - smoothstep(float(cur) - blur, float(cur) + blur, s);
        const float wn = smoothstep(float(cur + 1) - blur, float(cur + 1) + blur, s);

        const aten::v3 r = wc * c[cur] + wp * c[prv] + wn * c[nxt];
        return make_float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
    }
}
