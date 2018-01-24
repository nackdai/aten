#version 450
precision highp float;
precision highp int;

in vec3 normal;
in vec2 uv;
in vec3 baryCentric;
in float depth;
in vec4 prevCSPos;
flat in ivec2 ids;	// x: objid, y: primid.

uniform vec4 invScreen;

// NOTE
// x : objid
// y : primid
// zw : bary centroid
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outMotionDepth;

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
	vec2 curScreenPos = gl_FragCoord.xy * invScreen.xy;

	vec2 prevScreenPos = prevCSPos.xy / prevCSPos.w;
	prevScreenPos *= vec2(0.5, 0.5) + vec2(0.5);

	// [-1, 1] -> [0, 1]
	prevScreenPos = prevScreenPos * vec2(0.5) + vec2(0.5);

	// ([0, width], [0, height])
	vec2 motion = prevScreenPos - curScreenPos;
	motion /= invScreen.xy;

	outColor.x = intBitsToFloat(ids.x);	// objid
	outColor.y = intBitsToFloat(ids.y);	// primid
	outColor.zw = baryCentric.xy;

	outMotionDepth.x = motion.x;
	outMotionDepth.y = motion.y;
	outMotionDepth.z = depth;
	outMotionDepth.w = 1;
}
