#pragma once

#include <vector>
#include <functional>

#include "types.h"
#include "math/aabb.h"
#include "math/vec3.h"
#include "material/material.h"
#include "sampler/sampler.h"
#include "scene/context.h"

//#define ENABLE_TANGENTCOORD_IN_HITREC

namespace aten {
    class hitable;
    class accelerator;

    struct hitrecord {
        vec3 p;
        real area{ real(1) };

        vec3 normal;
        int32_t mtrlid{ -1 };

#ifdef ENABLE_TANGENTCOORD_IN_HITREC
        // tangent coordinate.
        vec3 du;
        vec3 dv;
#endif

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
                int32_t primid;
                real a, b;  // barycentric
                int32_t face;   // for cube.
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
            primid = -1;
            a = b = real(0);
            face = -1;

            nml_x = nml_y = nml_z = real(0);
            isVoxel = 0;
        }
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

        virtual int32_t geomid() const
        {
            return -1;
        }

        virtual accelerator* getInternalAccelerator();

        struct SamplePosNormalPdfResult {
            aten::vec3 pos;
            aten::vec3 nml;
            real area;

            real a;
            real b;
            int32_t primid{ -1 };
        };

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
            const Intersection& isect)
        {
            if (isect.isVoxel) {
                // For voxel.

                // Compute hit point.
                rec.p = r.org + isect.t * r.dir;
                rec.p = rec.p + AT_MATH_EPSILON * rec.normal;

                rec.normal.x = isect.nml_x;
                rec.normal.y = isect.nml_y;
                rec.normal.z = isect.nml_z;

                rec.mtrlid = isect.mtrlid;

                rec.isVoxel = true;
            }
            else {
                obj->evalHitResult(ctxt, r, rec, isect);
                rec.mtrlid = isect.mtrlid;
                rec.meshid = isect.meshid;

                rec.isVoxel = false;
            }

#ifdef ENABLE_TANGENTCOORD_IN_HITREC
            // tangent coordinate.
            rec.du = normalize(getOrthoVector(rec.normal));
            rec.dv = normalize(cross(rec.normal, rec.du));
#endif
        }

        static void evalHitResultForAreaLight(
            const context& ctxt,
            const hitable* obj,
            const ray& r,
            hitrecord& rec,
            const Intersection& isect)
        {
            obj->evalHitResult(ctxt, r, rec, isect);
            rec.mtrlid = isect.mtrlid;
            rec.meshid = isect.meshid;

#ifdef ENABLE_TANGENTCOORD_IN_HITREC
            // tangent coordinate.
            rec.du = normalize(getOrthoVector(rec.normal));
            rec.dv = normalize(cross(rec.normal, rec.du));
#endif
        }

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
