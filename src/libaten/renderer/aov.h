#pragma once

#include <array>

#include "defs.h"
#include "math/vec4.h"

namespace AT_NAME {
    struct AOVBufferType {
        enum Type {
            NormalDepth,
            AlbedoMeshId,
            end_of_AOVBufferType = AlbedoMeshId,
        } type_{ Type::NormalDepth };

        static constexpr int32_t NumBasicAovBuffer = static_cast<int32_t>(Type::end_of_AOVBufferType + 1);
        static constexpr int32_t BeginOfInheritType = static_cast<int32_t>(Type::end_of_AOVBufferType + 1);

        AT_DEVICE_API AOVBufferType() = default;
        AT_DEVICE_API ~AOVBufferType() = default;
        AT_DEVICE_API AOVBufferType(int32_t type) : type_(static_cast<Type>(type)) {}

        AT_DEVICE_API Type type() const { return type_; }

        AT_DEVICE_API friend bool operator==(const AOVBufferType& lhs, const AOVBufferType& rhs) {
            return lhs.type() == rhs.type();
        }
        AT_DEVICE_API friend bool operator!=(const AOVBufferType& lhs, const AOVBufferType& rhs) {
            return lhs.type() != rhs.type();
        }
    };

    struct AOVType {
        enum Type {
            Normal,
            Depth,
            Albedo,
            MeshId,
            end_of_AOVType = MeshId,
        } type_{ Type::Normal };

        static constexpr int32_t BeginOfInheritType = static_cast<int32_t>(Type::end_of_AOVType + 1);

        AT_DEVICE_API AOVType() = default;
        AT_DEVICE_API ~AOVType() = default;
        AT_DEVICE_API AOVType(int32_t type) : type_(static_cast<Type>(type)) {}

        AT_DEVICE_API Type type() const { return type_; }

        AT_DEVICE_API friend bool operator==(const AOVType& lhs, const AOVType& rhs) {
            return lhs.type() == rhs.type();
        }
        AT_DEVICE_API friend bool operator!=(const AOVType& lhs, const AOVType& rhs) {
            return lhs.type() != rhs.type();
        }
    };
}

namespace AT_NAME
{
    template <typename BUFFER_TYPE, size_t N>
    class AOVHostBuffer {
    public:
        static_assert(N > 0, "Empty buffer is not allowed");

        using buffer_type = BUFFER_TYPE;
        using buffer_value_type = typename BUFFER_TYPE::value_type;
        static constexpr auto IsEnoughBufferSizeForAlbedoMeshId = (N > AT_NAME::AOVBufferType::AlbedoMeshId);
        static constexpr auto NumAOV = N;

        AOVHostBuffer() = default;
        ~AOVHostBuffer() = default;

        AOVHostBuffer(const AOVHostBuffer&) = delete;
        AOVHostBuffer(AOVHostBuffer&&) = delete;
        AOVHostBuffer& operator=(const AOVHostBuffer&) = delete;
        AOVHostBuffer& operator=(AOVHostBuffer&&) = delete;

        template <int32_t M>
        buffer_type& get()
        {
            static_assert(M < N, "Over access AOV buffer");
            return aovs_[M];
        }

        buffer_type& normal_depth()
        {
            return get<static_cast<int32_t>(AT_NAME::AOVBufferType::NormalDepth)>();
        }

        [[nodiscard]] auto albedo_meshid() -> std::conditional_t<IsEnoughBufferSizeForAlbedoMeshId, buffer_type&, void>
        {
            if constexpr (IsEnoughBufferSizeForAlbedoMeshId) {
                return get<static_cast<int32_t>(AT_NAME::AOVBufferType::AlbedoMeshId)>();
            }
            else {
                return;
            }
        }

        void traverse(std::function<void(buffer_type&)> func)
        {
            for (auto& aov : aovs_) {
                func(aov);
            }
        }

    protected:
        std::array<buffer_type, N> aovs_;
    };

    template <typename BUFFER_VALUE_TYPE, typename TNormal, typename TAlbedo>
    inline AT_DEVICE_API void FillBasicAOVs(
        BUFFER_VALUE_TYPE& aovNormalDepth,
        const TNormal& normal,
        const aten::hitrecord& rec,
        const aten::mat4& mtxW2C,
        BUFFER_VALUE_TYPE& aovAlbedoMeshId,
        const TAlbedo& albedo,
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
        aovAlbedoMeshId.w = static_cast<real>(isect.meshid);
    }

    template <typename BUFFER_VALUE_TYPE, typename TBg>
    inline AT_DEVICE_API void FillBasicAOVsIfHitMiss(
        BUFFER_VALUE_TYPE& aovNormalDepth,
        BUFFER_VALUE_TYPE& aovAlbedoMeshId,
        const TBg& bg)
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

    template <typename BUFFER_VALUE_TYPE>
    inline AT_DEVICE_API void FillBaryCentricAOV(
        BUFFER_VALUE_TYPE& aovBuffer,
        const aten::Intersection& isect)
    {
        aovBuffer.x = isect.a;
        aovBuffer.y = isect.b;
        aovBuffer.z = real(1) - isect.a - isect.b;
    }

    template <typename BUFFER_VALUE_TYPE>
    inline AT_DEVICE_API void FillBaryCentricAOVIfHitMiss(BUFFER_VALUE_TYPE& aovBuffer)
    {
        aovBuffer.x = real(0);
        aovBuffer.y = real(0);
        aovBuffer.z = real(0);
    }
}
