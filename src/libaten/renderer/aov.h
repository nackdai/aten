#pragma once

#include <array>

#include "defs.h"
#include "math/vec4.h"

namespace aten
{
    struct AOVType {
        static constexpr int NormalDepth{ 0 };
        static constexpr int AlbedoMeshId{ 1 };
        static constexpr int NumBasicAov{ 2 };
    };

    template <template <typename T> typename Buffer, int N>
    class AOVHostBuffer {
    public:
        static_assert(N > 0);

        static constexpr auto IsEnoughBufferSizeForAlbedoMeshId = (N > AOVType::AlbedoMeshId);
        static constexpr auto NumAOV = N;
        using BufferType = typename Buffer<vec4>;

        AOVHostBuffer() = default;
        ~AOVHostBuffer() = default;

        AOVHostBuffer(const AOVHostBuffer&) = delete;
        AOVHostBuffer(AOVHostBuffer&&) = delete;
        AOVHostBuffer& operator=(const AOVHostBuffer&) = delete;
        AOVHostBuffer& operator=(AOVHostBuffer&&) = delete;

        template <int M>
        BufferType& get()
        {
            static_assert(M < N);
            return aovs_[M];
        }

        BufferType& normal_depth()
        {
            return get<static_cast<int>(AOVType::NormalDepth)>();
        }

        [[nodiscard]] auto albedo_meshid() -> std::conditional_t<IsEnoughBufferSizeForAlbedoMeshId, BufferType&, void>
        {
            if constexpr (IsEnoughBufferSizeForAlbedoMeshId) {
                return get<static_cast<int>(AOVType::AlbedoMeshId)>();
            }
            else {
                return;
            }
        }

        void traverse(std::function<void(BufferType&)> func)
        {
            for (auto& aov : aovs_) {
                func(aov);
            }
        }

    protected:
        std::array<BufferType, N> aovs_;
    };

    template <typename BufferType, typename DataType>
    inline AT_DEVICE_API void FillBasicAOVs(
        BufferType& aovNormalDepth,
        const DataType& normal,
        const aten::hitrecord& rec,
        const aten::mat4& mtxW2C,
        BufferType& aovAlbedoMeshId,
        const DataType& albedo,
        const aten::Intersection& isect)
    {
        // World coordinate to Clip coordinate.
        aten::vec4 pos(rec.p, 1);
        pos = mtxW2C.apply(pos);

        aovNormalDepth.x = normal.x;
        aovNormalDepth.y = normal.y;
        aovNormalDepth.z = normal.z;
        aovNormalDepth.w = pos.w;

        aovAlbedoMeshId.x = albedo.x;
        aovAlbedoMeshId.y = albedo.y;
        aovAlbedoMeshId.z = albedo.z;
        aovAlbedoMeshId.w = isect.meshid;
    }

    template <typename BufferType, typename DataType>
    inline AT_DEVICE_API void FillBasicAOVsIfHitMiss(
        BufferType& aovNormalDepth,
        BufferType& aovAlbedoMeshId,
        const DataType& bg)
    {
        aovNormalDepth.x = real(0);
        aovNormalDepth.y = real(0);
        aovNormalDepth.z = real(0);
        aovNormalDepth.w = -1;

        aovAlbedoMeshId.x = bg.x;
        aovAlbedoMeshId.y = bg.y;
        aovAlbedoMeshId.z = bg.z;
        aovAlbedoMeshId.w = -1;
    }

    template <typename BufferType>
    inline AT_DEVICE_API void FillBaryCentricAOV(
        BufferType& aovBuffer,
        const aten::Intersection& isect)
    {
        aovBuffer.x = isect.a;
        aovBuffer.y = isect.b;
        aovBuffer.z = real(1) - isect.a - isect.b;
    }

    template <typename BufferType>
    inline AT_DEVICE_API void FillBaryCentricAOVIfHitMiss(BufferType& aovBuffer)
    {
        aovBuffer.x = real(0);
        aovBuffer.y = real(0);
        aovBuffer.z = real(0);
    }
}
