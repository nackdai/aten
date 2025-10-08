#pragma once

#include <GL/glew.h>

#include "visualizer/atengl.h"
#include "visualizer/visualizer.h"
#include "math/vec2.h"

class Magnifier : public aten::visualizer::PostProc {
public:
    Magnifier() = default;
    virtual ~Magnifier() = default;

    Magnifier(const Magnifier&) = delete;
    Magnifier(Magnifier&&) = delete;
    Magnifier& operator=(const Magnifier&) = delete;
    Magnifier& operator=(Magnifier&&) = delete;

public:
    virtual void PrepareRender(
        const void* pixels,
        bool revert) override
    {
        shader::PrepareRender(pixels, revert);

        auto handle = GetHandle("image");
        if (handle >= 0) {
            CALL_GL_API(glUniform1i(handle, 0));
        }

        handle = GetHandle("screen_res");
        if (handle >= 0) {
            CALL_GL_API(glUniform2fv(handle, 1, reinterpret_cast<const GLfloat*>(&screen_res_)));
        }

        handle = GetHandle("center_pos");
        if (handle >= 0) {
            CALL_GL_API(glUniform2fv(handle, 1, reinterpret_cast<const GLfloat*>(&center_pos_)));
        }

        handle = GetHandle("magnification");
        if (handle >= 0) {
            CALL_GL_API(glUniform1f(handle, magnification_));
        }

        handle = GetHandle("radius");
        if (handle >= 0) {
            CALL_GL_API(glUniform1f(handle, radius_));
        }

        handle = GetHandle("circle_line_width");
        if (handle >= 0) {
            CALL_GL_API(glUniform1f(handle, circle_line_width_));
        }

        handle = GetHandle("circle_line_color");
        if (handle >= 0) {
            CALL_GL_API(glUniform3fv(handle, 1, reinterpret_cast<const GLfloat*>(&circle_line_color_)));
        }
    }

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
        screen_res_.x = getOutWidth();
        screen_res_.y = getOutHeight();

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
