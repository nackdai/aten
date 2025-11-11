#pragma once

#include "defs.h"
#include "vec3.h"
#include "vec4.h"

#ifdef __CUDACC__
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#endif

namespace aten
{
#ifdef __AT_CUDA__
    using v4 = float4;
    using v3 = float3;
#else
    using v4 = aten::vec4;
    using v3 = aten::vec3;
#endif

#ifndef __CUDACC__
    inline aten::vec3 make_float3(float x, float y, float z) { return { x, y, z }; }
    inline aten::vec4 make_float4(float x, float y, float z, float w) { return { x, y, z, w }; }
#endif

    template <class A, class B>
    inline AT_DEVICE_API void AddVec3(A& dst, const B& add)
    {
        if constexpr (std::is_same_v<A, B> && std::is_same_v<A, aten::vec3>) {
            dst += add;
        }
        else {
            dst += make_float3(add.x, add.y, add.z);
        }
    }

    // NOTE:
    // If template type A doesn't have the member variable "w", we can deal with it as vector 3 type.
    // Otherwise, we can deal with it as vector 4 type.

    template <class T>
    using HasMemberWOp = decltype(std::declval<T>().w);

    template <class A, class B>
    inline AT_DEVICE_API auto CopyVec(A& dst, const B& src)
        -> std::enable_if_t<!aten::is_detected<HasMemberWOp, A>::value, void>
    {
        if constexpr (std::is_same_v<A, B> && std::is_same_v<A, aten::vec3>) {
            dst = src;
        }
        else {
            dst = make_float3(src.x, src.y, src.z);
        }
    }

    template <class A, class B>
    inline AT_DEVICE_API auto CopyVec(A& dst, const B& src)
        -> std::enable_if_t<aten::is_detected<HasMemberWOp, A>::value, void>
    {
        if constexpr (std::is_same_v<A, B> && std::is_same_v<A, aten::vec4>) {
            dst = src;
        }
        else {
            dst = make_float4(src.x, src.y, src.z, src.w);
        }
    }

    template <class T = aten::v3>
    inline AT_DEVICE_API T MakeVec3(float x, float y, float z)
    {
        if constexpr (std::is_same_v<T, aten::vec3>) {
            return { x, y, z };
        }
        else {
            return make_float3(x, y, z);
        }
    }

    template <class T = aten::v4>
    inline AT_DEVICE_API T MakeVec4(float x, float y, float z, float w)
    {
        if constexpr (std::is_same_v<T, aten::vec4>) {
            return { x, y, z, w };
        }
        else {
            return make_float4(x, y, z, w);
        }
    }
}
