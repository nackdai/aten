#pragma once

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "accelerator/GpuPayloadDefs.h"
#include "geometry/geomparam.h"
#include "geometry/vertex.h"
#include "light/light_parameter.h"
#include "material/material.h"
#include "math/mat4.h"
#include "misc/tuple.h"
#include "misc/type_traits.h"
#include "image/texture.h"
#include "visualizer/GeomDataBuffer.h"

namespace AT_NAME {
    class triangle;
    class Light;
    class scene;
    class Grid;
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
        * @brief Control if alpha blending is enabled.
        */
        bool enable_alpha_blending{ false };

        /**
         * @brief Get the vertex position by index.
         * @note w specifies u of uv cooridnate of the vertex.
         * @param[in] idx Index to the vertex position.
         * @return Position of the vertex
         */
        const aten::vec4 GetPosition(uint32_t idx) const noexcept
        {
            const auto& v = GetVertex(idx);
            return aten::vec4(v.pos.x, v.pos.y, v.pos.z, v.uv.x);
        }

        /**
         * @brief Get the vertex position by index as aten::vec4 explicitly.
         * @param[in] idx Index to the vertex position.
         * @return Position of the vertex as aten::vec4.
         */
        const aten::vec4 GetPositionAsVec4(uint32_t idx) const noexcept
        {
            return GetPosition(idx);
        }

        /**
         * @brief Get the vertex position by index as aten::vec3 explicitly.
         * @param[in] idx Index to the vertex position.
         * @return Position of the vertex as aten::vec3.
         */
        const aten::vec3 GetPositionAsVec3(uint32_t idx) const noexcept
        {
            const auto& v = GetVertex(idx);
            return aten::vec3(v.pos.x, v.pos.y, v.pos.z);
        }

        /**
         * @brief Get the vertex normal by index.
         * @note w specifies v of uv cooridnate of the vertex.
         * @param[in] idx Index to the vertex normal.
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
         * @param[in] idx Index to the vertex normal.
         * @return Normal of the vertex as aten::vec4.
         */
        const aten::vec4 GetNormalAsVec4(uint32_t idx) const noexcept
        {
            return GetNormal(idx);
        }

        /**
         * @brief Get the vertex normal by index as aten::vec3 explicitly.
         * @param[in] idx Index to the vertex normal.
         * @return Normal of the vertex as aten::vec3.
         */
        const aten::vec3 GetNormalAsVec3(uint32_t idx) const noexcept
        {
            const auto& v = GetVertex(idx);
            return v.nml;
        }

        /**
         * @brief Get the object parameter by index.
         * @param[in] idx Index to the object parameter.
         * @return Object parameter.
         */
        const aten::ObjectParameter& GetObject(uint32_t idx) const noexcept;

        /**
         * @brief Get the list of object parameter and the list of the tranform matrix.
         * @return Tuple for the list of object parameter and the list of the tranform matrix.
         */
        aten::tuple<std::vector<aten::ObjectParameter>, std::vector<aten::mat4>> GetObjectParametersAndMatrices() const;

        /**
         * @brief Get number of registered objects.
         * @return Number of registered objects.
         */
        size_t GetObjectNum() const
        {
            return transformables_.size();
        }

        /**
         * @brief Get the material parameter by index.
         * @param[in] idx Index to the material parameter.
         * @return Material parameter.
         */
        const aten::MaterialParameter& GetMaterial(uint32_t idx) const noexcept
        {
            return GetMaterialInstance(idx)->param();
        }

        /**
         * @brief Get the list of the material paremeter.
         * @return List of the material paremeter.
         */
        std::vector<aten::MaterialParameter> GetMetarialParemeters() const;

        /**
         * @brief Get number of registered materials.
         * @return Number of registered materials.
         */
        size_t GetMaterialNum() const noexcept
        {
            return materials_.size();
        }

        /**
         * @brief Get the volumetric grid holder.
         * @return Voumetric grid holder.
         */
        const AT_NAME::Grid* GetGrid() const noexcept;

        void RegisterGridHolder(const std::shared_ptr<AT_NAME::Grid>& grid_holder);

        /**
         * @brief Get the triangle parameter by index.
         *
         * @param idx Index to the triangle parameter.
         * @return Triangle parameter.
         */
        const aten::TriangleParameter& GetTriangle(uint32_t idx) const noexcept;

        /**
         * @brief Get the light parameter by index.
         * @param[in] idx Index to the triangle parameter.
         * @return Triangle parameter.
         */
        const aten::LightParameter& GetLight(uint32_t idx) const noexcept;

        /**
         * @brief Get the list of the light parameter.
         * @return List of the light parameter.
         */
        std::vector<aten::LightParameter> GetLightParameters() const;

        /**
         * @brief Clear all registered lights.
         */
        void ClearAllLights()
        {
            lights_.clear();
        }

        /**
         * @brief Get the matrix by index.
         * @param[in] idx Index to the matrix.
         * @return Matrix.
         */
        const aten::mat4& GetMatrix(uint32_t idx) const noexcept
        {
            return *matrices_[idx];
        }

        /**
         * @brief Add a vertex information in the scene.
         * @param[in] vtx Vertex to be added.
         */
        void AddVertex(const aten::vertex& vtx)
        {
            vertices_.push_back(vtx);
        }

        /**
         * @brief Get the vertex by index.
         * @return Vertex.
         */
        const aten::vertex& GetVertex(int32_t idx) const
        {
            return vertices_[idx];
        }

        /**
        * @brief Replace the specified vertex.
        * @param[in] idx Index to be replaced.
        * @param[in] vtx Vertex to replcae.
        */
        void ReplaceVertex(int32_t idx, const aten::vertex& vtx)
        {
            vertices_[idx] = vtx;
        }

        /**
         * @brief Get the vertices as list.
         * @return Vertices as list.
         */
        const std::vector<aten::vertex>& GetVertices() const
        {
            return vertices_;
        }

        /**
         * @brief Get the list of extracted position and normal from the vertices.
         * @return List of extracted position and normal from the vertices.
         */
        aten::tuple<std::vector<aten::vec4>, std::vector<aten::vec4>> GetExtractedPosAndNmlInVertices() const;

        /**
         * @brief Get the number of the registered vertices.
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
         * @param[in] name Material name.
         * @param[in] type Material type.
         * @param[in] value Parameter value for the material.
         * @return Created material.
         */
        std::shared_ptr<AT_NAME::material> CreateMaterial(
            std::string_view name,
            aten::MaterialType type,
            aten::Values& value);

        /**
         * @brief Create a material with the default parameter value and then add it to the scene context.
         * @param[in] name Material name.
         * @param[in] param Material parameter.
         * @param[in] albedoMap Albedo map texture.
         * @param[in] normalMap Normal map texture.
         * @param[in] roughnessMap Roughness map texture.
         * @return Created material.
         */
        std::shared_ptr<AT_NAME::material> CreateMaterialWithMaterialParameter(
            std::string_view name,
            const aten::MaterialParameter& param,
            aten::texture* albedoMap,
            aten::texture* normalMap,
            aten::texture* roughnessMap);

        /**
         * @brief Get the actual material instance by index.
         * @param[in] idx Index to the actual material instance.
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
         * @param[in] dst Destination to copy registered materials' paramters.
         */
        void CopyMaterialParameters(std::vector<MaterialParameter>& dst) const;

        /**
         * @brief Find the material instance by name.
         * @param[in] name Name of the material instance.
         * @return If the material instance is found, return it. Otherwise, returns nullptr.
         */
        std::shared_ptr<AT_NAME::material> FindMaterialByName(std::string_view name) const
        {
            return GetAsset(materials_, name);
        }

        /**
         * @brief Find the index to the material instance by name.
         * @param[in] name Name of the material instance.
         * @return If the material instance is found, return the index to it. Otherwise, returns -1.
         */
        int32_t FindMaterialIdxByName(std::string_view name) const;

        /**
         * @brief Create a triangle instance.
         * @param[in] param Triangle parameter to create a triangle instance.
         * @param[in] scale Value to scale triangle and its vertices. If this is nullopt, nothing happens.
         * @return Created triangle instance.
         */
        std::shared_ptr<AT_NAME::triangle> CreateTriangle(
            const aten::TriangleParameter& param,
            std::optional<aten::vec3> scale = std::nullopt);

        /**
         * @brief Add a triangle instance to the scene.
         *
         * @param tri Triangle instance to be added.
         */
        void AddTriangle(const std::shared_ptr<AT_NAME::triangle>& tri);

        /**
         * @brief Get the number of all registered triangles in the scene.
         * @return Number of all registered triangles
         */
        uint32_t GetTriangleNum() const;

        /**
         * @brief Get the actual triangle instance by index.
         * @param[in] idx Index to the the actual triangle instance.
         * @return Triangle instance.
         */
        std::shared_ptr<AT_NAME::triangle> GetTriangleInstance(int32_t idx) const;

        /**
         * @brief Get the list of the triangle parameters.
         * @return List of the triangle parameters.
         */
        std::vector<aten::TriangleParameter> GetPrimitiveParameters() const;

        /**
         * @brief Copy the registered triangles' paramters to the specified list.
         * @param[out] dst Destination to copy registered triangles' paramters.
         */
        void CopyTriangleParameters(std::vector<aten::TriangleParameter>& dst) const;

        /**
         * @brief Check if the specified pointer is the registered triangle intance, and returns the index to it.
         * @param[in] p Pointer to be searched within the triangle instance list.
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
         * @brief[in] idx Index to the transformable instance
         * @return If the transformable instance is found, returns it. Otherwise, returns nullptr.
         */
        std::shared_ptr<aten::transformable> GetTransformable(int32_t idx) const;

        /**
         * @brief Traverse all registered transformable and execute the specified function for each transformable.
         * @param[in] func Function to be executed for each transformable.
         */
        void TraverseTransformables(
            std::function<void(std::shared_ptr<aten::transformable>&, aten::ObjectType)> func) const;

        /**
         * @brief Pick non identity matrices and update matrix index as -1 in transformable.
         * @return List to store all non identity matrices in the scene.
         */
        std::vector<aten::mat4> PickNonIdentityMatricesAndUpdateMatrixIdxInTransformable() const;

        /**
         * @brief Check if the specified pointer is the registered transformable intance, and returns the index to it.
         * @para[in]m p Pointer to be searched within the transformable instance list.
         * @return If the pointer is found as the registered transformable intance, returns the index to it. Otherwise, returns -1.
         */
        int32_t FindTransformableIdxFromPointer(const void* p) const;

        /**
         * @brief Check if the specified pointer is the registered transformable intance which is for polygon mesh object,
         *        and returns the order of only registered transformable intances which is for polygon mesh object.
         * @param[in] p Pointer to be searched within the transformable instance list.
         * @return If the  transformable intance which is for polygon mesh objec is found, returns its order. Otherwise, return -1
         */
        int32_t FindPolygonalTransformableOrderFromPointer(const void* p) const;

        /**
         * @brief Create a texture instance which is the empty and then add it to the scene context.
         * @param[in] width Width of the image data.
         * @param[in] height Height of the image data.
         * @param[in] channels Number of color channels.
         * @param[in] name Name of the texture instance.
         * @return Created texture instance.
         */
        std::shared_ptr<texture> CreateTexture(
            int32_t width, int32_t height,
            uint32_t channels,
            std::string_view name);

        /**
         * @brief Get the number of all registered texture instances.
         * @return Number of all registered texture instances.
         */
        auto GetTextureNum() const
        {
            return textures_.size();
        }

        /**
         * @brief Get the texture instance by index.
         * @param[in] idx Index to the texture instance.
         * @return If the texture instance is found, returns it. Otherwise, returns nullptr.
         */
        std::shared_ptr<texture> GetTexture(int32_t idx) const;

        /**
         * @brief Get the list to store the texture instances.
         * @return List to store the texture instances.
         */
        std::vector<std::shared_ptr<texture>> GetTextures();

        /**
         * @brief Initialize all registered texture instances as OpenGL texture.
         */
        void InitAllTextureAsGLTexture();

        /**
         * @brief Create an empty matrix and then add it to the scene context.
         * @return Tuple to store the index to the created matrix and the created matrix instance.
         */
        aten::tuple<uint32_t, std::shared_ptr<aten::mat4>> CreateMatrix()
        {
            size_t idx = matrices_.size();
            matrices_.reserve(idx + 1);
            matrices_.push_back(std::make_shared<aten::mat4>());
            return aten::make_tuple(static_cast<uint32_t>(idx), matrices_[idx]);
        }

        /**
         * @brief Add a light intance to the scene.
         * @param[in] light Light instance to be added.
         */
        void AddLight(const std::shared_ptr<AT_NAME::Light>& light);

        /**
         * @brief Get the actual light instance by index.
         * @param[in] idx Index to the light instance.
         * @return If the light instance is found, returns it. Otherwise, returns nullptr.
         */
        std::shared_ptr<AT_NAME::Light> GetLightInstance(uint32_t idx) const;

        /**
         * @brief Get the number of all registered light instances.
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

        void CleanAll();

        /**
         * @brief Get the texture instance by name.
         * @param[in] name Name of the texture instance.
         * @return If the texture instance is found, returns it. Otherwise, returns nullptr.
         */
        std::shared_ptr<texture> GetTextureByName(std::string_view name) const
        {
            return GetAsset(textures_, name);
        }

        void UpdateSceneBoundingBox(const AT_NAME::scene& scene);

        const aten::aabb& GetSceneBoundingBox() const
        {
            //AT_ASSERT(scene_bounding_box_.IsValid());
            return scene_bounding_box_;
        }

    private:
        /**
         * @brief Add a texture instance.
         * @param[in] tex Texture instance to be added.
         */
        void AddTexture(const std::shared_ptr<texture>& tex);

        /**
         * @brief Add a material to the scene.
         * @param[in] mtrl The material to be added.
         */
        void AddMaterial(std::shared_ptr<AT_NAME::material> mtrl)
        {
            AT_ASSERT(mtrl);
            materials_.push_back(mtrl);
            AT_ASSERT(materials_.size() < std::numeric_limits<decltype(MaterialParameter::id)>::max());
            mtrl->param().id = static_cast<uint16_t>(materials_.size() - 1);
        }

        /**
         * @brief Copy BVH nodes to the context.
         * @param[in] src Source BVH nodes.
         */
        void CopyBvhNodes(const std::vector<std::vector<GPUBvhNode>>& src)
        {
            nodes_ = src;
        }

        /**
         * @brief Get the BVH nodes by index.
         * @param[in] idx Index to the BVH nodes.
         * @return BVH nodes.
         */
        const std::vector<GPUBvhNode>& GetBvhNodes(size_t idx) const
        {
            return nodes_[idx];
        }

    private:
        std::vector<aten::vertex> vertices_;

        aten::GeomVertexBuffer vertex_buffer_;

        std::vector<std::shared_ptr<AT_NAME::material>> materials_;
        std::vector<std::shared_ptr<AT_NAME::triangle>> triangles_;
        mutable std::vector<std::shared_ptr<aten::transformable>> transformables_;
        std::vector<std::shared_ptr<aten::texture>> textures_;
        std::vector<std::shared_ptr<aten::mat4>> matrices_;
        std::vector<std::shared_ptr<AT_NAME::Light>> lights_;

        std::vector<std::vector<GPUBvhNode>> nodes_;

        // NOTE:
        // This variable for AT_NAME::Grid needs to be defined with the template.
        // Because, we do  forward declaration for AT_NAME::Grid not to include the header file directly in this file.
        std::shared_ptr<AT_NAME::Grid> grid_holder_;

        aten::aabb scene_bounding_box_;

        bool is_window_initialized_{ false };

        template <class T>
        static auto GetAsset(const std::vector<T>& assets, std::string_view name) -> std::enable_if_t<aten::is_shared_ptr_v<T>, T>
        {
            if (name.empty()) {
                return nullptr;
            }

            auto found = std::find_if(
                assets.begin(), assets.end(),
                [name](const auto& t) {
                    if (t->nameString() == name) {
                        return true;
                    }
                    return false;
                });
            if (found != assets.end()) {
                return *found;
            }
            return nullptr;
        }
    };
}
