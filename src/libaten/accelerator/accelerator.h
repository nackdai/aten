#pragma once

#include <memory>
#include <vector>

#include "scene/hitable.h"

namespace aten {
    class context;

    /**
     * @enum AccelType
     * @brief Enumulation for acceleration structures.
     */
    enum class AccelType {
        Bvh,            ///< BVH.
        Sbvh,           ///< SBVH.
        ThreadedBvh,    ///< Threaded BVH.
        StacklessBvh,   ///< Stackless BVH.
        UserDefs,       ///< User defined.

        Default,        ///< Default type.
    };

    /**
     * @brief Base class for acceleration structure.
     */
    class accelerator : public hitable {
        friend class PolygonObject;
        friend class deformable;
        template<class ACCEL> friend class AcceleratedScene;

    protected:
        accelerator(AccelType type) : accel_type_{ type } {}

    public:
        accelerator() = delete;
        virtual ~accelerator() = default;

    private:
        static AccelType s_internalType;
        static std::function<std::shared_ptr<accelerator>()> s_userDefsInternalAccelCreator;

        /**
         * @brief Return a created acceleration structure for internal used.
         */
        static std::shared_ptr<accelerator> createAccelerator(AccelType type = AccelType::Default);

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
            is_nested_ = true;
        }

    public:
        /**
         * @brief Set a function to create user defined acceleration structure for internal used.
         */
        static void setUserDefsInternalAccelCreator(
            std::function<std::shared_ptr<accelerator>()> creator);

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
        virtual bool HitWithLod(
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
            const aten::mat4& mtx_L2W)
        {
            AT_ASSERT(false);
        }

        struct ResultIntersectTestByFrustum {
            int32_t ep{ -1 };   ///< Entry Point.
            int32_t ex{ -1 };   ///< Layer Id.

            // 1つ上のレイヤーへの戻り先のノードID.
            int32_t top{ -1 };  ///< Upper layer id.

            int32_t padding{ 0 };

            ResultIntersectTestByFrustum() = default;
        };

        /**
         * @brief Export the built structure data.
         */
        virtual bool exportTree(
            const context& ctxt,
            std::string_view path)
        {
            AT_ASSERT(false);
            return false;
        }

        /**
         * @brief Import the exported structure data.
         */
        virtual bool importTree(
            const context& ctxt,
            std::string_view path,
            int32_t offsetTriIdx)
        {
            AT_ASSERT(false);
            return false;
        }

        /**
         * @brief Return the type about acceleration structure.
         */
        AccelType getAccelType()
        {
            return accel_type_;
        }

    protected:
        bool isExporting() const
        {
            return is_exporting_;
        }
        void enableExporting()
        {
            is_exporting_ = true;
        }

    protected:
        // Type about acceleration structure.
        AccelType accel_type_{ AccelType::Bvh };

        // Flag whether accelerator is nested.
        bool is_nested_{ false };

        // Flag whether accelerator is exporting structure data.
        bool is_exporting_{ false };
    };
}
