#pragma once

#include <array>
#include <functional>

#include "defs.h"
#include "math/vec4.h"
#include "misc/span.h"
#include "scene/hit_parameter.h"

namespace AT_NAME {
    /**
     * @brief Inheritable basic AOV buffer types.
     */
    struct AOVBufferType {
        enum Type {
            NormalDepth,    ///< Normal and depth.
            AlbedoMeshId,   ///< Albedo and mesh id.
            end_of_AOVBufferType = AlbedoMeshId,    ///< End of types.
        } type_{ Type::NormalDepth };

        /** Number of the basic AOV buffer types. */
        static constexpr int32_t NumBasicAovBuffer = static_cast<int32_t>(Type::end_of_AOVBufferType + 1);

        /** Enumuration value to begin in the inherited enumuration. */
        static constexpr int32_t BeginOfInheritType = static_cast<int32_t>(Type::end_of_AOVBufferType + 1);

        AOVBufferType() = default;
        ~AOVBufferType() = default;
        AT_HOST_DEVICE_API AOVBufferType(int32_t type) : type_(static_cast<Type>(type)) {}

        AT_HOST_DEVICE_API Type type() const { return type_; }

        AT_HOST_DEVICE_API friend bool operator==(const AOVBufferType& lhs, const AOVBufferType& rhs) {
            return lhs.type() == rhs.type();
        }
        AT_HOST_DEVICE_API friend bool operator!=(const AOVBufferType& lhs, const AOVBufferType& rhs) {
            return lhs.type() != rhs.type();
        }
    };

    /**
     * @brief Inheritable basic AOV types.
     */
    struct AOVType {
        enum Type {
            Normal, ///< Normal.
            Depth,  ///< Depth.
            Albedo, ///< Albedo.
            MeshId, ///< Mesh id.
            end_of_AOVType = MeshId,    ///< End of types.
        } type_{ Type::Normal };

        static constexpr int32_t BeginOfInheritType = static_cast<int32_t>(Type::end_of_AOVType + 1);

        AOVType() = default;
        ~AOVType() = default;
        AT_HOST_DEVICE_API AOVType(int32_t type) : type_(static_cast<Type>(type)) {}

        AT_HOST_DEVICE_API Type type() const { return type_; }

        AT_HOST_DEVICE_API friend bool operator==(const AOVType& lhs, const AOVType& rhs) {
            return lhs.type() == rhs.type();
        }
        AT_HOST_DEVICE_API friend bool operator!=(const AOVType& lhs, const AOVType& rhs) {
            return lhs.type() != rhs.type();
        }
    };
}

namespace AT_NAME
{
    /**
     * @brief AOV buffer in host.
     *
     * @tparam BUFFER_TYPE Container type for AOV buffer.
     * @tparam N Number of the managed AOV buffers.
     */
    template <class BUFFER_TYPE, size_t N>
    class AOVHostBuffer {
    public:
        static_assert(N > 0, "Empty buffer is not allowed");

        using buffer_type = BUFFER_TYPE;
        using buffer_value_type = typename BUFFER_TYPE::value_type;
        using buffer_type_as_span = aten::span<buffer_value_type>;
        using buffer_type_as_const_span = aten::const_span<buffer_value_type>;

        /** Whether enough number of the managed AOV buffers is specified to store the buffer for albedo and mesh id. */
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

        template <int32_t M>
        buffer_type_as_span GetAsSpan()
        {
            auto& aov = get<M>();
            return buffer_type_as_span(aov.data(), aov.size());
        }

        template <int32_t M>
        buffer_type_as_const_span GetAsConstSpan()
        {
            auto& aov = get<M>();
            return buffer_type_as_const_span(aov.data(), aov.size());
        }

        buffer_type& normal_depth()
        {
            return get<static_cast<int32_t>(AT_NAME::AOVBufferType::NormalDepth)>();
        }

        buffer_type_as_span GetNormalDepthAsSpan()
        {
            return GetAsSpan<static_cast<int32_t>(AT_NAME::AOVBufferType::NormalDepth)>();
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

        buffer_type_as_span GetAlbedoMeshIdAsSpan()
        {
            return GetAsSpan<static_cast<int32_t>(AT_NAME::AOVBufferType::AlbedoMeshId)>();
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

    template <class BUFFER_VALUE_TYPE, class TNormal, class TAlbedo>
    inline AT_HOST_DEVICE_API void FillBasicAOVs(
        BUFFER_VALUE_TYPE& aovNormalDepth,
        const TNormal& normal,
        const aten::hitrecord& rec,
        const aten::mat4& mtx_W2C,
        BUFFER_VALUE_TYPE& aovAlbedoMeshId,
        const TAlbedo& albedo,
        const aten::Intersection& isect)
    {
        // World coordinate to Clip coordinate.
        aten::vec4 pos(rec.p, 1);
        pos = mtx_W2C.apply(pos);

        aovNormalDepth.x = normal.x;
        aovNormalDepth.y = normal.y;
        aovNormalDepth.z = normal.z;
        aovNormalDepth.w = pos.w;

        aovAlbedoMeshId.x = albedo.x;
        aovAlbedoMeshId.y = albedo.y;
        aovAlbedoMeshId.z = albedo.z;
        aovAlbedoMeshId.w = static_cast<float>(isect.meshid);
    }

    template <class BUFFER_VALUE_TYPE, class TBg>
    inline AT_HOST_DEVICE_API void FillBasicAOVsIfHitMiss(
        BUFFER_VALUE_TYPE& aovNormalDepth,
        BUFFER_VALUE_TYPE& aovAlbedoMeshId,
        const TBg& bg)
    {
        aovNormalDepth.x = float(0);
        aovNormalDepth.y = float(0);
        aovNormalDepth.z = float(0);
        aovNormalDepth.w = -1;

        aovAlbedoMeshId.x = bg.x;
        aovAlbedoMeshId.y = bg.y;
        aovAlbedoMeshId.z = bg.z;
        aovAlbedoMeshId.w = -1;
    }

    template <class BUFFER_VALUE_TYPE>
    inline AT_HOST_DEVICE_API void FillBaryCentricAOV(
        BUFFER_VALUE_TYPE& aovBuffer,
        const aten::Intersection& isect)
    {
        aovBuffer.x = isect.hit.tri.a;
        aovBuffer.y = isect.hit.tri.b;
        aovBuffer.z = float(1) - isect.hit.tri.a - isect.hit.tri.b;
    }

    template <class BUFFER_VALUE_TYPE>
    inline AT_HOST_DEVICE_API void FillBaryCentricAOVIfHitMiss(BUFFER_VALUE_TYPE& aovBuffer)
    {
        aovBuffer.x = float(0);
        aovBuffer.y = float(0);
        aovBuffer.z = float(0);
    }
}
