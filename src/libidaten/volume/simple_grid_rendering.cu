#include <filesystem>

#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/cuda/CudaDeviceBuffer.h>

#include "volume/volume_rendering.h"

#include "kernel/device_scene_context.cuh"
#include "kernel/intersect.cuh"
#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "renderer/pathtracing/pt_params.h"
#include "camera/pinhole.h"

namespace idaten
{
    class VolumeRendering::SimpleGridRenderer {
    public:
        SimpleGridRenderer() = default;
        ~SimpleGridRenderer() = default;

        bool LoadNanoVDB(std::string_view nvdb, cudaStream_t stream);

        void Render(
            cudaSurfaceObject_t dst,
            cudaStream_t stream,
            int32_t width, int32_t height,
            const aten::CameraParameter& camera,
            const aten::vec3& bg_color);

        bool IsLoaded() const
        {
            return gpu_grid_;
        }

        const aten::aabb& GetBoundingBox() const
        {
            return grid_bbox_;
        }

    private:
        nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> handle_;
        const nanovdb::FloatGrid* gpu_grid_{ nullptr };
        aten::aabb grid_bbox_;
        bool is_density_grid_{ false };
    };

    bool VolumeRendering::SimpleGridRenderer::LoadNanoVDB(std::string_view nvdb, cudaStream_t stream)
    {
        std::filesystem::path p = nvdb;

        if (!std::filesystem::exists(p)) {
            AT_ASSERT(false);
            AT_PRINTF("%s doesn't exist.", nvdb.data());
            return false;
        }

        try {
            auto list = nanovdb::io::readGridMetaData(nvdb.data());
            if (list.size() != 1) {
                // TODO
                // Support only one grid.
                AT_PRINTF("Support only one grid\n");
                return false;
            }

            is_density_grid_ = list[0].gridName == "density";

            handle_ = nanovdb::io::readGrid<nanovdb::CudaDeviceBuffer>(nvdb.data());
            handle_.deviceUpload(stream, true);
            gpu_grid_ = handle_.deviceGrid<float>();
            if (!gpu_grid_) {
                AT_PRINTF("GridHandle did not contain a grid with value type float\n");
                return false;
            }
            auto cpu_grid = handle_.grid<float>();
            const auto nvdb_grid_bbox = cpu_grid->worldBBox();
            grid_bbox_.init(
                aten::vec3(nvdb_grid_bbox.min()[0], nvdb_grid_bbox.min()[1], nvdb_grid_bbox.min()[2]),
                aten::vec3(nvdb_grid_bbox.max()[0], nvdb_grid_bbox.max()[1], nvdb_grid_bbox.max()[2])
            );
            return true;
        }
        catch (const std::exception& e) {
            AT_PRINTF("An exception occurred: %s\n", e.what());
            return false;
        }
    }

    __global__ void RenderSimpleGrid(
        cudaSurfaceObject_t dst,
        const nanovdb::FloatGrid* grid,
        bool is_density_grid,
        int32_t width, int32_t height,
        const aten::CameraParameter camera,
        const aten::vec3 bg_color)
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        auto color = make_float4(bg_color.x, bg_color.y, bg_color.z, 1.0F);

#if 1
        // Get accessor.
        auto accessor = grid->tree().getAccessor();

        using Vec3F = nanovdb::Vec3<float>;
        using RayF = nanovdb::Ray<float>;

        // Generate path.
        const auto s = static_cast<float>(x) / width;
        const auto t = static_cast<float>(y) / height;

        AT_NAME::CameraSampleResult cam_sample;
        AT_NAME::PinholeCamera::sample(&cam_sample, &camera, s, t);

        RayF world_ray(
            Vec3F(cam_sample.r.org.x, cam_sample.r.org.y, cam_sample.r.org.z),
            Vec3F(cam_sample.r.dir.x, cam_sample.r.dir.y, cam_sample.r.dir.z));

        RayF index_ray = world_ray.worldToIndexF(*grid);

        if (is_density_grid) {
            const auto tree_index_bbox = grid->tree().bbox();

            // Clip to bounds.
            if (index_ray.clip(tree_index_bbox)) {
                // Integrate.

                // TODO
                constexpr float dt = 0.5F;

                float transmittance = 1.0f;
                for (float t = index_ray.t0(); t < index_ray.t1(); t += dt) {
                    const auto sigma = accessor.getValue(nanovdb::Coord::Floor(index_ray(t))) * 0.1f;
                    transmittance *= 1.0F - sigma * dt;
                }

                const auto tr = 1.0F - transmittance;
                color = make_float4(tr, tr, tr, 1.0F);
            }
        }
        else {
            // Intersect.
            float  t0{ 0.0F };
            nanovdb::Coord ijk;
            float  v{ 0.0F };

            if (nanovdb::ZeroCrossing(index_ray, accessor, ijk, v, t0)) {
                // Render distance to surface as the uniform color with the assumption it is a uniform voxel.
                float wT0 = t0 * static_cast<float>(grid->voxelSize()[0]);
                //float t = (wT0 - camera.znear) / (camera.zfar - camera.znear);
                color = make_float4(wT0, wT0, wT0, 1.0F);
            }
        }
#endif

        surf2Dwrite(
            color,
            dst,
            x * sizeof(float4), y,
            cudaBoundaryModeTrap);
    }

    void VolumeRendering::SimpleGridRenderer::Render(
        cudaSurfaceObject_t dst,
        cudaStream_t stream,
        int32_t width, int32_t height,
        const aten::CameraParameter& camera,
        const aten::vec3& bg_color)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        RenderSimpleGrid << <grid, block, 0, stream >> > (
            dst,
            gpu_grid_, is_density_grid_,
            width, height, camera, bg_color);

        checkCudaKernel(RenderSimpleGrid);
    }

    std::optional<aten::aabb> VolumeRendering::LoadNanoVDB(std::string_view nvdb)
    {
        if (!simple_grid_renderer_) {
            simple_grid_renderer_ = std::make_shared<SimpleGridRenderer>();
        }

        if (!simple_grid_renderer_->IsLoaded()) {
            if (!simple_grid_renderer_->LoadNanoVDB(nvdb, m_stream)) {
                return std::nullopt;
            }
        }

        const auto grid_bbox = simple_grid_renderer_->GetBoundingBox();
        return grid_bbox;
    }

    void VolumeRendering::RenderNanoVDB(
        GLuint gltex,
        int32_t width, int32_t height,
        const aten::CameraParameter& camera,
        const aten::vec3 bg_color/*= aten::vec3(0.0F, 0.5F, 1.0F)*/)
    {
        if (!m_stream) {
            cudaStreamCreate(&m_stream);
        }

        if (!m_glimg.IsValid()) {
            m_glimg.init(gltex, idaten::CudaGLRscRegisterType::ReadWrite);
        }

        CudaGLResourceMapper<decltype(m_glimg)> rscmap(m_glimg);
        auto output_surface = m_glimg.bind();

        simple_grid_renderer_->Render(
            output_surface, m_stream,
            width, height,
            camera,
            bg_color);
    }
}
