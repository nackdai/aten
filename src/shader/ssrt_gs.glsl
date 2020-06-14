#version 450

precision highp float;
precision highp int;

layout(triangles) in;
layout(triangle_strip, max_vertices = 6) out;

uniform mat4 mtxW2C;
uniform mat4 mtxPrevW2C;
uniform int objid;
uniform int primid;

in vec3 worldNormal[3];
in vec2 vUV[3];
in vec4 prevWorldPos[3];

out vec3 normal;
out vec2 uv;
out vec3 baryCentric;
out float depth;
out vec4 prevCSPos;
out vec4 curCSPos;
flat out ivec2 ids;

// TODO
// パストレ内の計算方法と合うように順番を対応させているので汎用性に関しては要検討.

// For computing bary centric.
const vec3 weight[3] = {
    vec3(0, 0, 1),
    vec3(1, 0, 0),
    vec3(0, 1, 0),
};

void main()
{
    for (int i = 0; i < gl_in.length(); i++) {
        gl_Position = mtxW2C * gl_in[i].gl_Position;

        curCSPos = gl_Position;
        prevCSPos = mtxPrevW2C * prevWorldPos[i];

        depth = gl_Position.w;

        normal = worldNormal[i];
        uv = vUV[i];
        baryCentric = weight[i];

        ids.x = objid;
        ids.y = primid + gl_PrimitiveIDIn;

        EmitVertex();
    }
    EndPrimitive();
}
