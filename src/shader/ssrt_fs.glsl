#version 450
precision highp float;
precision highp int;

in vec3 normal;
in vec2 uv;
in vec3 baryCentric;
flat in ivec2 ids;	// x: objid, y: primid.

layout(location = 0) out vec4 outColor;

const vec3 clr[8] = {
	vec3(0, 0, 0),
	vec3(0, 0, 1),
	vec3(0, 1, 0),
	vec3(0, 1, 1),
	vec3(1, 0, 0),
	vec3(1, 0, 1),
	vec3(0, 1, 1),
	vec3(1, 1, 1),
};

void main()
{
	//outColor.xyz = normal * 0.5 + 0.5;
	
	//outColor.xyz = baryCentric;

	outColor.xyz = clr[ids.y % 8];

	outColor.w = 1.0;
}
