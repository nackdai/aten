#pragma once

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "geometry/vertex.h"
#include "texture/texture.h"
#include "visualizer/GeomDataBuffer.h"
#include "material/material.h"
#include "light/light_parameter.h"
#include "geometry/geomparam.h"
#include "math/mat4.h"

namespace AT_NAME {
    class triangle;
    class Light;
}

namespace aten
{
    class transformable;

    /**
     * @brief Scene context.
     */
    class context {
    public:
        context() = default;
        ~context() = default;

        context(const context&) = delete;
        context(context&&) = delete;
        context& operator=(const context&) = delete;
        context& operator=(context&&) = delete;

        /**
         * @brief Get the vertex position by index.
         *
         * @note w specifies u of uv cooridnate of the vertex.
         * @param idx Index to the vertex position.
         * @return Position of the vertex
         */
        const aten::vec4 GetPosition(uint32_t idx) const noexcept
        {
            const auto& v = GetVertex(idx);
            return aten::vec4(v.pos.x, v.pos.y, v.pos.z, v.uv.x);
        }

        /**
         * @brief Get the vertex position by index as aten::vec4 explicitly.
         *
         * @param idx Index to the vertex position.
         * @return Position of the vertex as aten::vec4.
         */
        const aten::vec4 GetPositionAsVec4(uint32_t idx) const noexcept
        {
            return GetPosition(idx);
        }

        /**
         * @brief Get the vertex position by index as aten::vec3 explicitly.
         *
         * @param idx Index to the vertex position.
         * @return Position of the vertex as aten::vec3.
         */
        const aten::vec3 GetPositionAsVec3(uint32_t idx) const noexcept
        {
            const auto& v = GetVertex(idx);
            return aten::vec3(v.pos.x, v.pos.y, v.pos.z);
        }

        /**
         * @brief Get the vertex normal by index.
         *
         * @note w specifies v of uv cooridnate of the vertex.
         * @param idx Index to the vertex normal.
         * @return Normal of the vertex
         */
        const aten::vec4 GetNormal(uint32_t idx) const noexcept
        {
            const auto& v = GetVertex(idx);
            aten::vec4 res(v.nml, v.uv.y);
            return res;
        }

        /**
         * @brief Get the vertex normal by index as aten::vec4 explicitly.
         *
         * @param idx Index to the vertex normal.
         * @return Normal of the vertex as aten::vec4.
         */
        const aten::vec4 GetNormalAsVec4(uint32_t idx) const noexcept
        {
            return GetNormal(idx);
        }

        /**
         * @brief Get the vertex normal by index as aten::vec3 explicitly.
         *
         * @param idx Index to the vertex normal.
         * @return Normal of the vertex as aten::vec3.
         */
        const aten::vec3 GetNormalAsVec3(uint32_t idx) const noexcept
        {
            const auto& v = GetVertex(idx);
            return v.nml;
        }

        /**
         * @brief Get the object parameter by index.
         *
         * @param idx Index to the object parameter.
         * @return Object parameter.
         */
        const aten::ObjectParameter& GetObject(uint32_t idx) const noexcept;

        /**
         * @brief Get the material parameter by index.
         *
         * @param idx Index to the material parameter.
         * @return Material parameter.
         */
        const aten::MaterialParameter& GetMaterial(uint32_t idx) const noexcept
        {
            return GetMaterialInstance(idx)->param();
        }

        /**
         * @brief Get the triangle parameter by index.
         *
         * @param idx Index to the triangle parameter.
         * @return Triangle parameter.
         */
        const aten::TriangleParameter& GetTriangle(uint32_t idx) const noexcept;

        /**
         * @brief Get the triangle parameter by index.
         * @param idx Index to the triangle parameter.
         * @return Triangle parameter.
         */
        const aten::LightParameter& GetLight(uint32_t idx) const noexcept;

        /**
         * @brief Get the matrix by index.
         *
         * @param idx Index to the matrix.
         * @return Matrix.
         */
        const aten::mat4& GetMatrix(uint32_t idx) const noexcept
        {
            return *matrices_[idx];
        }

        /**
         * @brief Add a vertex information in the scene.
         *
         * @param vtx Vertex to be added.
         */
        void AddVertex(const aten::vertex& vtx)
        {
            vertices_.push_back(vtx);
        }

        /**
         * @brief Get the vertex by index.
         *
         * @return Vertex.
         */
        const aten::vertex& GetVertex(int32_t idx) const
        {
            return vertices_[idx];
        }

        /**
         * @brief Get the number of the registered vertices.
         *
         * @return Number of the registered vertices.
         */
        uint32_t GetVertexNum() const
        {
            return (uint32_t)vertices_.size();
        }

        /**
         * @brief Copy the registered vertices to the specified list.
         */
        void CopyVertices(std::vector<vertex>& dst) const
        {
            std::copy(
                vertices_.begin(),
                vertices_.end(),
                std::back_inserter(dst));
        }

        void build();

        /**
         * @brief Get the vertex buffer of the scene.
         *
         * @return Vertex buffer for the scene.
         */
        const aten::GeomVertexBuffer& GetVertexBuffer() const
        {
            return vertex_buffer_;
        }

        void release()
        {
            vertices_.clear();
            vertex_buffer_.clear();
        }

        /**
         * @brief Create a material and then add it to the scene context.
         *
         * @param type Material type.
         * @param value Parameter value for the material.
         * @return Created material.
         */
        std::shared_ptr<AT_NAME::material> CreateMaterial(
            aten::MaterialType type,
            aten::Values& value);

        /**
         * @brief Create a material with the default parameter value and then add it to the scene context.
         *
         * @param type Material type.
         * @return Created material.
         */
        std::shared_ptr<AT_NAME::material> CreateMaterialWithDefaultValue(
            aten::MaterialType type);

        /**
         * @brief Create a material with the default parameter value and then add it to the scene context.
         *
         * @param param Material parameter.
         * @param albedoMap Albedo map texture.
         * @param normalMap Normal map texture.
         * @param roughnessMap Roughness map texture.
         * @return Created material.
         */
        std::shared_ptr<AT_NAME::material> CreateMaterialWithMaterialParameter(
            const aten::MaterialParameter& param,
            aten::texture* albedoMap,
            aten::texture* normalMap,
            aten::texture* roughnessMap);

        /**
         * @brief Add a material to the scene.
         *
         * @param mtrl The material to be added.
         */
        void AddMaterial(std::shared_ptr<AT_NAME::material> mtrl)
        {
            AT_ASSERT(mtrl);
            materials_.push_back(mtrl);
            AT_ASSERT(materials_.size() < std::numeric_limits<decltype(MaterialParameter::id)>::max());
            mtrl->param().id = static_cast<uint16_t>(materials_.size() - 1);
        }

        /**
         * @brief Get the actual material instance by index.
         *
         * @param idx Index to the actual material instance.
         * @return Actual material instance.
         */
        std::shared_ptr<AT_NAME::material> GetMaterialInstance(int32_t idx) const
        {
            AT_ASSERT(0 <= idx && idx < static_cast<int32_t>(materials_.size()));
            return materials_[idx];
        }

        /**
         * @brief Delete all registered materials and clear the material list.
         */
        void DeleteAllMaterialsAndClearList();

        /**
         * @brief Copy the registered materials' paramters to the specified list.
         *
         * @param dst Destination to copy registered materials' paramters.
         */
        void CopyMaterialParameters(std::vector<MaterialParameter>& dst) const;

        /**
         * @brief Find the material instance by name.
         *
         * @param name Name of the material instance.
         * @return If the material instance is found, return it. Otherwise, returns nullptr.
         */
        std::shared_ptr<const AT_NAME::material> FindMaterialByName(std::string_view name) const;

        /**
         * @brief Find the index to the material instance by name.
         *
         * @param name Name of the material instance.
         * @return If the material instance is found, return the index to it. Otherwise, returns -1.
         */
        int32_t FindMaterialIdxByName(std::string_view name) const;

        /**
         * @brief Create a triangle instance.
         *
         * @param param Triangle parameter to create a triangle instance.
         * @return Created triangle instance.
         */
        std::shared_ptr<AT_NAME::triangle> CreateTriangle(const aten::TriangleParameter& param);

        /**
         * @brief Add a triangle instance to the scene.
         *
         * @param tri Triangle instance to be added.
         */
        void AddTriangle(const std::shared_ptr<AT_NAME::triangle>& tri);

        /**
         * @brief Get the number of all registered triangles in the scene.
         *
         * @return Number of all registered triangles
         */
        uint32_t GetTriangleNum() const;

        /**
         * @brief Get the actual triangle instance by index.
         *
         * @param idx Index to the the actual triangle instance.
         * @return Triangle instance.
         */
        std::shared_ptr<const AT_NAME::triangle> GetTriangleInstance(int32_t idx) const;

        /**
         * @brief Copy the registered triangles' paramters to the specified list.
         *
         * @param dst Destination to copy registered triangles' paramters.
         */
        void CopyTriangleParameters(std::vector<aten::TriangleParameter>& dst) const;

        /**
         * @brief Check if the specified pointer is the registered triangle intance, and returns the index to it.
         *
         * @param p Pointer to be searched within the triangle instance list.
         * @return If the pointer is found as the registered triangle intance, returns the index to it. Otherwise, returns -1.
         */
        int32_t FindTriangleIdxFromPointer(const void* p) const;

        /**
         * @brief Add a transformable instance.
         *
         * @param t Transformable instance to be added.
         */
        void AddTransformable(const std::shared_ptr<transformable>& t);

        /**
         * @brief Get the transformable instance by index.
         *
         * @brief idx Index to the transformable instance
         * @return If the transformable instance is found, returns it. Otherwise, returns nullptr.
         */
        std::shared_ptr<const aten::transformable> GetTransformable(int32_t idx) const;

        /**
         * @brief Traverse all registered transformable and execute the specified function for each transformable.
         *
         * @param func Function to be executed for each transformable.
         */
        void TraverseTransformables(
            std::function<void(std::shared_ptr<aten::transformable>&, aten::ObjectType)> func) const;

        /**
         * @brief Pick non identity matrices and update matrix index as -1 in transformable.
         *
         * @return List to store all non identity matrices in the scene.
         */
        std::vector<aten::mat4> PickNonIdentityMatricesAndUpdateMatrixIdxInTransformable() const;

        /**
         * @brief Check if the specified pointer is the registered transformable intance, and returns the index to it.
         *
         * @param p Pointer to be searched within the transformable instance list.
         * @return If the pointer is found as the registered transformable intance, returns the index to it. Otherwise, returns -1.
         */
        int32_t FindTransformableIdxFromPointer(const void* p) const;

        /**
         * @brief Check if the specified pointer is the registered transformable intance which is for polygon mesh object,
         *        and returns the order of only registered transformable intances which is for polygon mesh object.
         *
         * @param p Pointer to be searched within the transformable instance list.
         * @return If the  transformable intance which is for polygon mesh objec is found, returns its order. Otherwise, return -1
         */
        int32_t FindPolygonalTransformableOrderFromPointer(const void* p) const;

        /**
         * @brief Create a texture instance which is the empty and then add it to the scene context.
         *
         * @param width Width of the image data.
         * @param height Height of the image data.
         * @param channels Number of color channels.
         * @param name Name of the texture instance.
         * @return Created texture instance.
         */
        std::shared_ptr<texture> CreateTexture(
            int32_t width, int32_t height,
            uint32_t channels,
            std::string_view name);

        /**
         * @brief Get the number of all registered texture instances.
         *
         * @return Number of all registered texture instances.
         */
        auto GetTextureNum() const
        {
            return textures_.size();
        }

        /**
         * @brief Get the texture instance by index.
         *
         * @param idx Index to the texture instance.
         * @return If the texture instance is found, returns it. Otherwise, returns nullptr.
         */
        std::shared_ptr<texture> GtTexture(int32_t idx) const;

        /**
         * @brief Add a texture instance.
         *
         * @param tex Texture instance to be added.
         */
        void AddTexture(const std::shared_ptr<texture>& tex);

        /**
         * @brief Initialize all registered texture instances as OpenGL texture.
         */
        void InitAllTextureAsGLTexture();


        static void PinContext(const context* ctxt)
        {
            s_pinnedCtxt = ctxt;
        }

        static void removePinnedContext()
        {
            s_pinnedCtxt = nullptr;
        }

        static const context* getPinnedContext()
        {
            AT_ASSERT(s_pinnedCtxt);
            return s_pinnedCtxt;
        }

        /**
         * @brief Create an empty matrix and then add it to the scene context.
         *
         * @return Tuple to store the index to the created matrix and the created matrix instance.
         */
        std::tuple<uint32_t, std::shared_ptr<aten::mat4>> CreateMatrix()
        {
            size_t idx = matrices_.size();
            matrices_.reserve(idx + 1);
            matrices_.push_back(std::make_shared<aten::mat4>());
            return std::make_tuple(static_cast<uint32_t>(idx), matrices_[idx]);
        }

        /**
         * @brief Add a light intance to the scene.
         *
         * @param ligh Light instance to be added.
         */
        void AddLight(const std::shared_ptr<AT_NAME::Light>& light);

        /**
         * @brief Get the actual light instance by index.
         *
         * @param idx Index to the light instance.
         * @return If the light instance is found, returns it. Otherwise, returns nullptr.
         */
        std::shared_ptr<AT_NAME::Light> GetLightInstance(uint32_t idx) const;

        /**
         * @brief Get the number of all registered light instances.
         *
         * @return Number of all registered light instances.
         */
        size_t GetLightNum() const;

        void SetIsWindowInitialized(bool b)
        {
            is_window_initialized_ = b;
        }

        bool IsWindowInitialized() const
        {
            return is_window_initialized_;
        }

    private:
        static const context* s_pinnedCtxt;

        std::vector<aten::vertex> vertices_;

        aten::GeomVertexBuffer vertex_buffer_;

        std::vector<std::shared_ptr<AT_NAME::material>> materials_;
        std::vector<std::shared_ptr<AT_NAME::triangle>> triangles_;
        mutable std::vector<std::shared_ptr<aten::transformable>> transformables_;
        std::vector<std::shared_ptr<aten::texture>> textures_;
        std::vector<std::shared_ptr<aten::mat4>> matrices_;
        std::vector<std::shared_ptr<AT_NAME::Light>> lights_;

        bool is_window_initialized_{ false };
    };
}
