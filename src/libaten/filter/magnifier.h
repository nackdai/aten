#pragma once

#include "visualizer/visualizer.h"
#include "math/vec2.h"

namespace aten {
    class Magnifier : public visualizer::PostProc {
    public:
        Magnifier() = default;
        virtual ~Magnifier() = default;

        Magnifier(const Magnifier&) = delete;
        Magnifier(Magnifier&&) = delete;
        Magnifier& operator=(const Magnifier&) = delete;
        Magnifier& operator=(Magnifier&&) = delete;

    public:
        virtual void prepareRender(
            const void* pixels,
            bool revert) override;

        virtual aten::PixelFormat inFormat() const override
        {
            return aten::PixelFormat::rgba8;
        }
        virtual aten::PixelFormat outFormat() const override
        {
            return aten::PixelFormat::rgba8;
        }

        virtual void setParam(aten::Values& values) override final
        {
            screen_res_.x = static_cast<real>(getOutWidth());
            screen_res_.y = static_cast<real>(getOutHeight());

            const auto center_pos = values.get("center_pos", aten::vec4(0));
            center_pos_.x = center_pos.x;
            center_pos_.y = center_pos.y;

            magnification_ = values.get("magnification", magnification_);;
            radius_ = values.get("radius", radius_);;

            circle_line_width_ = values.get("circle_line_width", magnification_);;
            circle_line_color_ = values.get("circle_line_color", aten::vec4(0));;
        }

    private:
        aten::vec2 screen_res_;
        aten::vec2 center_pos_;
        float magnification_{ 0.0f };
        float radius_{ 0.0f };
        float circle_line_width_{ 2.0f };
        aten::vec3 circle_line_color_;
    };
}
