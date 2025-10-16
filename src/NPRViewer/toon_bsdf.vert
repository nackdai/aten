#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 normal;

uniform mat4 mtx_L2W;
uniform mat4 mtx_W2C;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outPosition;
layout(location = 2) out vec3 outTangent;
layout(location = 3) out vec3 outBinormal;

vec3 GetOrthoVector(vec3 n)
{
    vec3 p;
    float k = sqrt(n.y * n.y + n.z * n.z);
    p.x = 0;
    p.y = -n.z / k;
    p.z = n.y / k;
    return normalize(p);
}

void GetTangentCoordinate(in vec3 n, out vec3 t, out vec3 b)
{
    t = GetOrthoVector(n);
    b = cross(n, t);
    t = cross(b, n);
}

void main()
{
    gl_Position = mtx_W2C * mtx_L2W * position;

    outPosition = (mtx_L2W * position).xyz;
    vec3 n = normalize(mtx_L2W * vec4(normal, 0)).xyz;

    vec3 t, b;
    GetTangentCoordinate(n, t, b);

    outNormal = n;
    outTangent = b;
    outBinormal = t;
}
