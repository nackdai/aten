#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec2 uv;

uniform sampler2D s0;	// albedo.
uniform vec4 color;

uniform bool hasAlbedo = false;

uniform int materialId;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outAttrib;

void main()
{
	if (hasAlbedo) {
		outColor = texture2D(s0, uv) * color;
	}
	else {
		// TODO
		outColor = color;
	}

	// NOTE
	// 0 ‚Í‰½‚à‚È‚¢ˆµ‚¢‚É‚·‚é...
	outAttrib = vec4((materialId + 1) / 255.0f, 0, 0, 1);
}
