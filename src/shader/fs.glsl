#version 420
precision highp float;
precision highp int;

uniform sampler2D image;

uniform vec4 invScreen;
uniform bool revert;
uniform bool isRenderRGB;

// output colour for the fragment
layout(location = 0) out highp vec4 oColour;

void main()
{
    vec2 uv = gl_FragCoord.xy * invScreen.xy;
	if (revert) {
		uv.x = 1.0 - uv.x;
		uv.y = 1.0 - uv.y;
	}

    vec4 color = texture2D(image, uv);

	if (isRenderRGB) {
		oColour.rgb = color.rgb;
		oColour.a = 1;
	}
	else {
		oColour.rgb = vec3(color.a, color.a, color.a);
		oColour.a = 1;
	}
}
