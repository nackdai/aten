#pragma once

#include <vector>
#include "scene/hitable.h"
#include "math/frustum.h"
#include "scene/context.h"

namespace aten {
    /**
     * @enum AccelType
     * @brief Enumulation for acceleration structures.
     */
    enum class AccelType {
        Bvh,            ///< BVH.
        Qbvh,            ///< QBVH.
        Sbvh,            ///< SBVH.
        ThreadedBvh,    ///< Threaded BVH.
        StacklessBvh,    ///< Stackless BVH.
        StacklessQbvh,    ///< Stackless QBVH.
        UserDefs,        ///< User defined.

        Default,        ///< Default type.
    };

    /**
     * @brief Base class for acceleration structure.
     */
    class accelerator : public hitable {
        friend class object;
        friend class deformable;
        template<typename ACCEL> friend class AcceleratedScene;

    private:
        accelerator() {}

    protected:
        accelerator(AccelType type)
        {
            m_type = type;
        }
        virtual ~accelerator() {}

    private:
        static AccelType s_internalType;
        static std::function<accelerator*()> s_userDefsInternalAccelCreator;

        /**
         * @brief Return a created acceleration structure for internal used.
         */
        static accelerator* createAccelerator(AccelType type = AccelType::Default);

        /**
         * @brief Set the acceleration structure type for internal used.
         */
        static void setInternalAccelType(AccelType type);

        /**
         * @brief Return the acceleration structure type for internal used.
         */
        static AccelType getInternalAccelType();

        /**
         * @brief Specify that the acceleration structure is nested.
         */
        void asNested()
        {
            m_isNested = true;
        }

    public:
        /**
         * @brief Set a function to create user defined acceleration structure for internal used.
         */
        static void setUserDefsInternalAccelCreator(std::function<accelerator*()> creator);

        /**
         * @brief Bulid structure tree from the specified list.
         */
        virtual void build(
            const context& ctxt,
            hitable** list,
            uint32_t num,
            aabb* bbox) = 0;

        /**
         * @brief Build voxel data from the structure tree.
         */
        virtual void buildVoxel(const context& ctxt)
        {
            // Nothing is done.
            AT_ASSERT(false);
        }

        /**
         * @brief Test if a ray hits a object.
         */
        virtual bool hit(
            const context& ctxt,
            const ray& r,
            real t_min, real t_max,
            bool enableLod,
            Intersection& isect) const = 0;

        /**
         * @brief Update the structure tree.
         */
        virtual void update(const context& ctxt)
        {
            AT_ASSERT(false);
        }

        /**
         * @brief Draw all node's AABB in the structure tree.
         */
        virtual void drawAABB(
            aten::hitable::FuncDrawAABB func,
            const aten::mat4& mtxL2W)
        {
            AT_ASSERT(false);
        }

        struct ResultIntersectTestByFrustum {
            int ep{ -1 };    ///< Entry Point.
            int ex{ -1 };    ///< Layer Id.

            // 1つ上のレイヤーへの戻り先のノードID.
            int top{ -1 };    ///< Upper layer id.

            int padding;

            ResultIntersectTestByFrustum() {}
        };

        /**
         * @brief Export the built structure data.
         */
        virtual bool exportTree(
            const context& ctxt,
            const char* path)
        {
            AT_ASSERT(false);
            return false;
        }

        /**
         * @brief Import the exported structure data.
         */
        virtual bool importTree(
            const context& ctxt,
            const char* path, 
            int offsetTriIdx)
        {
            AT_ASSERT(false);
            return false;
        }

        /**
         * @brief Return the type about acceleration structure.
         */
        AccelType getAccelType()
        {
            return m_type;
        }

    protected:
        bool isExporting() const
        {
            return m_isExporting;
        }
        void enableExporting()
        {
            m_isExporting = true;
        }

    protected:
        // Type about acceleration structure.
        AccelType m_type{ AccelType::Bvh };

        // Flag whether accelerator is nested.
        bool m_isNested{ false };

        // Flag whether accelerator is exporting structure data.
        bool m_isExporting{ false };
    };
}
