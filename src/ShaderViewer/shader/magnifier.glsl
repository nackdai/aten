#version 420
precision highp float;
precision highp int32_t;

uniform sampler2D image;

// Screen resolution.
uniform vec2 screen_res;

// Target position as center of magnifilter.
uniform vec2 center_pos;

// Variable to describe how magnify. [0, 1)
uniform float magnification;

// Radius to adopt magnification.
uniform float radius;

// Line width of magnifier circle.
uniform float circle_line_width;

// Line color of magnifier circle.
uniform vec3 circle_line_color;

layout(location = 0) in vec2 uv;

// output colour for the fragment
layout(location = 0) out highp vec4 oColour;

void main()
{
    vec2 center_uv = center_pos / screen_res;

    // NOTE:
    // e.g. center_pos is center of screen.
    // (0,0)                  (1,0)
    //   +---------------------+
    //   |       (0.5,0.5)     |
    //   |          +     x    * (1,0.5)
    //   |           (0.75,0.5)|
    //   +---------------------+
    // (0,1)                  (1,1)
    //
    // magnification = 0.5
    // magnifiled_proportional_uv = mix(uv, center_uv, magnification) = uv * (1 - magnification) + magnification * center_uv
    //  => magnifiled_proportional_uv = (1, 0.5) * (1 - 0.5) + 0.5 * (0.5,0.5) = (0.75,0.5)

    // Compute proportional uv to magnify.
    vec2 magnifiled_uv = mix(uv, center_uv, magnification);

    vec2 frag_coord = gl_FragCoord.xy;
    frag_coord.y = screen_res.y - frag_coord.y;

    float d = length(frag_coord - center_pos);
    if (d > radius)
    {
        magnifiled_uv = uv;
    }

    if (radius < d && d <= (radius + circle_line_width)) {
        oColour.rgb = circle_line_color;
    }
    else {
        vec4 color = texture2D(image, magnifiled_uv);
        oColour.rgb = color.rgb;
    }

    oColour.a = 1;
}
