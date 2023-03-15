#pragma once

#include <vector>
#include <functional>

#include "types.h"
#include "math/aabb.h"
#include "math/vec3.h"
#include "material/material.h"
#include "sampler/sampler.h"

namespace aten {
    class context;
    class hitable;
    class accelerator;

    struct hitrecord {
        vec3 p;
        real area{ real(1) };

        vec3 normal;
        int32_t mtrlid{ -1 };

        // texture coordinate.
        real u{ real(0) };
        real v{ real(0) };

        int32_t meshid{ -1 };

        bool isVoxel{ false };
        uint8_t padding[3];
    };

    struct Intersection {
        real t{ AT_MATH_INF };

        int32_t objid{ -1 };

        int32_t mtrlid{ -1 };

        int32_t meshid{ -1 };

        union {
            // For triangle.
            struct {
                int32_t triangle_id;    ///< Hit triangle id.
                real a, b;              ///< Barycentric on hit triangle.
                real padding;
            };
            // Fox voxel.
            struct {
                real nml_x;
                real nml_y;
                real nml_z;
                int32_t isVoxel;
            };
        };

        AT_DEVICE_API Intersection()
        {
            triangle_id = -1;
            a = b = real(0);

            nml_x = nml_y = nml_z = real(0);
            isVoxel = 0;
        }
    };

    /**
    * @brief Result for sampled triangle.
    **/
    struct SamplePosNormalPdfResult {
        aten::vec3 pos; ///< Sampled position.
        aten::vec3 nml; ///< Normal at sampled position.
        real area;      ///< Area of sampled triangle.

        real a; ///< Barycentric on sampled triangle.
        real b; ///< Barycentric on sampled triangle.
        int32_t triangle_id{ -1 };  ///< Sampled triangle id.
    };

    using NotifyChanged = std::function<void(hitable*)>;

    class hitable {
    public:
        hitable(const char* name = nullptr)
        {
            if (name) {
                m_name = name;
            }
        }
        virtual ~hitable() {}

    public:
        virtual bool hit(
            const context& ctxt,
            const ray& r,
            real t_min, real t_max,
            Intersection& isect) const = 0;

        virtual const aabb& getBoundingbox() const
        {
            return m_aabb;
        }
        void setBoundingBox(const aabb& bbox)
        {
            m_aabb = bbox;
        }

        virtual aabb getTransformedBoundingBox() const
        {
            return m_aabb;
        }

        virtual const hitable* getHasObject() const
        {
            return nullptr;
        }

        virtual const hitable* getHasSecondObject() const
        {
            return nullptr;
        }

        bool isInstance() const
        {
            return (getHasObject() != nullptr);
        }

        virtual int32_t mesh_id() const
        {
            return -1;
        }

        virtual accelerator* getInternalAccelerator();

        virtual void getSamplePosNormalArea(
            const context& ctxt,
            SamplePosNormalPdfResult* result,
            sampler* sampler) const
        {
            AT_ASSERT(false);
        }

        static void evalHitResult(
            const context& ctxt,
            const hitable* obj,
            const ray& r,
            hitrecord& rec,
            const Intersection& isect);

        static void evalHitResultForAreaLight(
            const context& ctxt,
            const hitable* obj,
            const ray& r,
            hitrecord& rec,
            const Intersection& isect);

        using FuncPreDraw = std::function<void(const aten::mat4& mtxL2W, const aten::mat4& mtxPrevL2W, int32_t parentId, int32_t basePrimId)>;

        virtual void render(
            FuncPreDraw func,
            const context& ctxt,
            const aten::mat4& mtxL2W,
            const aten::mat4& mtxPrevL2W,
            int32_t parentId,
            uint32_t triOffset)
        {
            // For rasterize rendering.
            AT_ASSERT(false);
        }

        using FuncDrawAABB = std::function<void(const aten::mat4&)>;

        virtual void drawAABB(
            FuncDrawAABB func,
            const aten::mat4& mtxL2W)
        {
            // For debug rendering.
            AT_ASSERT(false);
        }

        void setFuncNotifyChanged(NotifyChanged onNotifyChanged)
        {
            m_onNotifyChanged = onNotifyChanged;
        }

        virtual void update()
        {
            // Nothing is done...
        }

        virtual uint32_t getTriangleCount() const
        {
            return 0;
        }

        virtual bool isDeformable() const
        {
            return false;
        }

        void setName(const char* name)
        {
            m_name = name;
        }
        const char* name()
        {
            return m_name.c_str();
        }

    protected:
        void onNotifyChanged()
        {
            if (m_onNotifyChanged) {
                m_onNotifyChanged(this);
            }
        }

    private:
        virtual void evalHitResult(
            const context& ctxt,
            const ray& r,
            hitrecord& rec,
            const Intersection& isect) const
        {
            AT_ASSERT(false);
        }

    private:
        std::string m_name;
        aabb m_aabb;

        NotifyChanged m_onNotifyChanged;
    };
}
