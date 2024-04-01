#pragma once

#include "visualizer/pixelformat.h"
#include "visualizer/visualizer.h"

namespace aten {
    class FlakesNormalMapMaker : public visualizer::PostProc {
    public:
        FlakesNormalMapMaker() {}
        virtual ~FlakesNormalMapMaker() {}

    public:
        virtual void prepareRender(
            const void* pixels,
            bool revert) override;

        virtual PixelFormat inFormat() const override
        {
            return PixelFormat::rgba8;
        }
        virtual PixelFormat outFormat() const override
        {
            return PixelFormat::rgba8;
        }

        struct Parameter {
            float flake_scale{ 50.0f };                // Smaller values zoom into the flake map, larger values zoom out.
            float flake_size{ 0.5f };                // Relative size of the flakes
            float flake_size_variance{ 0.7f };        // 0.0 makes all flakes the same size, 1.0 assigns random size between 0 and the given flake size
            float flake_normal_orientation{ 0.5f };    // Blend between the flake normals (0.0) and the surface normal (1.0)

            Parameter() {}
            Parameter(
                float scale,
                float size,
                float size_var,
                float orient)
            {
                flake_scale = scale;
                flake_size = size;
                flake_size_variance = scale;
                flake_normal_orientation = orient;
            }
        };

        const Parameter& GetParameter() const
        {
            return param_;
        }

        Parameter& GetParameter()
        {
            return param_;
        }

    private:
        Parameter param_;
    };

}
