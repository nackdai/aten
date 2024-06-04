#pragma once

#include <stack>

#include "material/material.h"
#include "math/vec3.h"
#include "math/vec4.h"
#include "misc/stack.h"
#include "sampler/sampler.h"

#ifdef __AT_CUDA__
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#endif

namespace AT_NAME {
    namespace _detail {
#ifdef __AT_CUDA__
        using v4 = float4;
        using v3 = float3;
#else
        using v4 = aten::vec4;
        using v3 = aten::vec3;
#endif
    };

    struct PathThroughput {
        aten::vec3 throughput;
        float pdfb;

        static constexpr size_t MedisumStackSize = 8;
        aten::stack<int32_t, MedisumStackSize> mediums;
        //std::stack<int32_t> mediums;
    };

    struct PathContrib {
        union {
            _detail::v4 v;
            struct {
                _detail::v3 contrib;
                float samples;
            };
        };
#ifndef __AT_CUDA__
        PathContrib() : contrib(0), samples(0.0f) {}

        PathContrib(const PathContrib& rhs)
        {
            v = rhs.v;
        }
        PathContrib(PathContrib&& rhs) noexcept
        {
            v = rhs.v;
        }
        PathContrib& operator=(const PathContrib& rhs)
        {
            v = rhs.v;
            return *this;
        }
        PathContrib& operator=(PathContrib&& rhs) noexcept
        {
            v = rhs.v;
            return *this;
        }
        ~PathContrib() = default;
#endif
    };

    struct PathAttribute {
        bool isHit{ false };
        bool isTerminate{ false };
        bool isSingular{ false };
        bool isKill{ false };

        aten::MaterialType mtrlType{ aten::MaterialType::Lambert };
    };

    struct Path {
        PathThroughput* throughput;
        PathContrib* contrib;
        PathAttribute* attrib;
        aten::sampler* sampler;
    };

    struct PathHost {
        Path paths;

        bool init(int32_t width, int32_t height)
        {
            if (throughput.empty()) {
                throughput.resize(width * height);
                contrib.resize(width * height);
                attrib.resize(width * height);
                sampler.resize(width * height);

                paths.throughput = throughput.data();
                paths.contrib = contrib.data();
                paths.attrib = attrib.data();
                paths.sampler = sampler.data();

                return true;
            }
            return false;
        }

        void Clear(int32_t frame_cnt, std::function<void(void*, int32_t, size_t)> clear)
        {
            clear(throughput.data(), 0, sizeof(decltype(throughput)::value_type) * throughput.size());
            clear(contrib.data(), 0, sizeof(decltype(contrib)::value_type) * contrib.size());
            clear(attrib.data(), 0, sizeof(decltype(attrib)::value_type) * attrib.size());

            // NOTE:
            // sampler should be kept parmanetly while rendering.
            // For CUDA, the container for sampler is not the normal std container.
            // And, the sampler element's constructor is not called.
            // It means initialization doesn't happen and the value of variables are undefined.
            // So, we need to clear by filling with zero. But, it should be done once at the first frame.
            // The following has to be executed for only CUDA.
            if constexpr (!std::is_same_v<decltype(sampler), std::vector<aten::sampler>>) {
                if (frame_cnt == 0) {
                    clear(sampler.data(), 0, sizeof(decltype(sampler)::value_type) * sampler.size());
                }
            }
        }

        template <class ClearFunc, typename ExtraArgToClearFunc = void>
        void Clear(int32_t frame_cnt, ClearFunc clear, ExtraArgToClearFunc extra_arg)
        {
            Clear(
                frame_cnt,
                [clear, extra_arg](void* dst, int32_t val, size_t size) { clear(dst, val, size, extra_arg); });
        }

        void Clear(int32_t frame_cnt)
        {
            Clear(frame_cnt, memset);
        }

#ifdef __AT_CUDA__
        idaten::TypedCudaMemory<PathThroughput> throughput;
        idaten::TypedCudaMemory<PathContrib> contrib;
        idaten::TypedCudaMemory<PathAttribute> attrib;
        idaten::TypedCudaMemory<aten::sampler> sampler;
#else
        std::vector<PathThroughput> throughput;
        std::vector<PathContrib> contrib;
        std::vector<PathAttribute> attrib;
        std::vector<aten::sampler> sampler;
#endif
    };

    struct ShadowRay {
        aten::vec3 rayorg;
        float distToLight;

        aten::vec3 raydir;
        struct {
            uint32_t isActive : 1;
        };

        aten::vec3 lightcontrib;
        uint32_t targetLightId;
    };

    /**
     * @brief Matrices for rendering.
     */
    struct MatricesForRendering {
        aten::mat4 mtx_W2V;         ///< Matrix to convert from World coordinate to View cooridnate.
        aten::mat4 mtx_V2C;         ///< Matrix to convert from View coordinate to Clip cooridnate.
        aten::mat4 mtx_C2V;         ///< Matrix to convert from Clip coordinate to View cooridnate.

        aten::mat4 mtx_V2W;         ///< Matrix to convert from View coordinate to World cooridnate.
        aten::mat4 mtx_PrevW2V;     ///< Matrix to convert from World coordinate to View cooridnate in the previous frame.

        /**
         * @param Get a matrix to convert from World coordinate to Clip cooridnate.
         *
         * @return Matrix to convert from World coordinate to Clip cooridnate.
         */
        aten::mat4 GetW2C() const
        {
            return mtx_V2C * mtx_W2V;
        }

        /**
         * @brief Reset the matrices with the specified camera parameter.
         *
         * @param[in] camera Camera parameter to reset the matrices.
         */
        void Reset(const aten::CameraParameter& camera)
        {
            mtx_PrevW2V = mtx_W2V;

            camera::ComputeCameraMatrices(camera, mtx_W2V, mtx_V2C);

            mtx_C2V = mtx_V2C;
            mtx_C2V.invert();

            mtx_V2W = mtx_W2V;
            mtx_V2W.invert();
        }
    };
}
