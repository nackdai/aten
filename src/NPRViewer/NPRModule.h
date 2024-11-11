#pragma once

#include "aten.h"

class NPRModule {
public:
    NPRModule() = default;
    virtual ~NPRModule() {}

    virtual bool Init(
        int32_t width, int32_t height,
        std::string_view path_vs,
        std::string_view path_fs)
    {
        return rasterizer_.init(width, height, path_vs, path_fs);
    }

    virtual void Draw(
        aten::context& ctxt,
        std::vector<std::shared_ptr<aten::PolygonObject>>& objs,
        aten::PinholeCamera& camera,
        bool is_wireframe
    )
    {
        rasterizer_.drawWithOutsideRenderFunc(
            ctxt,
            [&](aten::RasterizeRenderer::FuncObjRenderer func) {
                auto& shader = rasterizer_.getShader();
                PreRender(shader, camera);

                for (size_t i = 0; i < objs.size(); i++) {
                    auto& obj = objs[i];
                    func(*obj);
                }
            },
            &camera, is_wireframe);
    }

    virtual void InitDebugVisual(
        int32_t width, int32_t height,
        std::string_view path_vs,
        std::string_view path_fs)
    {}

    virtual void PreRender(aten::shader& shader, const aten::PinholeCamera& camera)
    {}

    virtual void DrawDebugVisual(
        const aten::context& ctxt,
        const aten::Camera& cam)
    {}

    virtual void EditParameter()
    {}

    aten::RasterizeRenderer& rasterizer()
    {
        return rasterizer_;
    }

 protected:
    aten::RasterizeRenderer rasterizer_;
};
