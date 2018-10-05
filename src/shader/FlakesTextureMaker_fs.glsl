#version 450
precision highp float;
precision highp int;

uniform vec4 u_resolution;
uniform float u_time;

#if 0
vec2 hash(vec2 p)
{
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p)*18.5453);
}

#define K (6.2831)

// return distance, and cell id
vec2 voronoi(in vec2 x)
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    float flake_size = 0.05;
    vec3 m = vec3(flake_size);

    bool isValid = false;

    for (int j = -1; j <= 1; j++)
    {
        for (int i = -1; i <= 1; i++)
        {
            vec2  g = vec2(float(i), float(j));
            vec2  o = hash(n + g);
            vec2  r = g - f + (0.5 + 0.5 * sin(u_time + K * o));

            // distance.
            float d = dot(r, r);

            if (d < m.x) {
                m = vec3(d, o);
                isValid = true;
            }
        }
    }
    return isValid ? vec2(sqrt(m.x), m.y + m.z) : vec2(0, 0);
}

void main()
{
    vec2 ndcPos = gl_FragCoord.xy / u_resolution.xy;
    ndcPos = ndcPos * 2.0 - 1.0;    // [0, 1] -> [-1, 1]

    float flake_scale = 1;
    vec2 p = ndcPos * 1.0 / flake_scale;
    
    vec2 c = voronoi((14.0 + 6.0 * sin(0.2 * u_time)) * p);

    vec3 nv = vec3(0, 0, 1);

    if (dot(c, c) > 0) {
        float x = 0.5 + 0.5 * cos(c.y * K);
        float y = 0.5 + 0.5 * sin(c.y * K);

        // calculate normal vector
        nv = normalize(vec3(x * 2.0 - 1.0, y * 2.0 - 1.0, 1.0));

        float flake_orientation = 0.1;
        nv = mix(nv, vec3(0, 0, 1), flake_orientation);
        nv = normalize(nv);
    }

    // encode normal vector
    gl_FragColor = vec4(nv * 0.5 + 0.5, 1.0);
}
#else
float bits_to_01(uint bits)
{
    // divide by 2^32-1
    uint div = 0xffffffff;
    return bits * (1.0 / float(div));
}

uint rotl32(uint var, uint hops)
{
    return (var << hops) | (var >> (32 - hops));
}

// Bob Jenkins "lookup3" hashes:  http://burtleburtle.net/bob/c/lookup3.c
// It's in the public domain.

// Mix up the bits of a, b, and c (changing their values in place).
void bjmix(inout uint a, inout uint b, inout uint c)
{
    a -= c;  a ^= rotl32(c, 4);  c += b;
    b -= a;  b ^= rotl32(a, 6);  a += c;
    c -= b;  c ^= rotl32(b, 8);  b += a;
    a -= c;  a ^= rotl32(c, 16);  c += b;
    b -= a;  b ^= rotl32(a, 19);  a += c;
    c -= b;  c ^= rotl32(b, 4);  b += a;
}

// Mix up and combine the bits of a, b, and c (doesn't change them, but
// returns a hash of those three original values).  21 ops
uint bjfinal(uint a, uint b, uint c)
{
    c ^= b; c -= rotl32(b, 14);
    a ^= c; a -= rotl32(c, 11);
    b ^= a; b -= rotl32(a, 25);
    c ^= b; c -= rotl32(b, 16);
    a ^= c; a -= rotl32(c, 4);
    b ^= a; b -= rotl32(a, 14);
    c ^= b; c -= rotl32(b, 24);
    return c;
}

uint inthash(uvec4 k)
{
    int N = 4;

    // now hash the data!
    uint len = N;
    uint a = 0xdeadbeef + (len << 2) + 13;
    uint b = 0xdeadbeef + (len << 2) + 13;
    uint c = 0xdeadbeef + (len << 2) + 13;

#if 0
    while (len > 3) {
        a += k[0];
        b += k[1];
        c += k[2];
        bjmix(a, b, c);
        len -= 3;

        // NOTE
        // オリジナルコードは、k[N]の想定で、ポインタの加算を行っている.
        k += 3;
    }

    switch (len) {
    case 3: c += k[2];
    case 2: b += k[1];
    case 1: a += k[0];
        c = bjfinal(a, b, c);
    case 0:
        break;
    }
#else
    a += k[0];
    b += k[1];
    c += k[2];
    bjmix(a, b, c);

    a += k[3];
    c = bjfinal(a, b, c);
#endif

    return c;
}

vec3 hash3(uvec4 k)
{
    int N = 4;

    vec3 result;

    k[N - 1] = 0;
    result.x = bits_to_01(inthash(k));

    k[N - 1] = 1;
    result.y = bits_to_01(inthash(k));

    k[N - 1] = 2;
    result.z = bits_to_01(inthash(k));

    return result;
}

vec3 cellnoise(vec3 p)
{
    uvec4 iv;
    iv[0] = uint(floor(p.x));
    iv[1] = uint(floor(p.y));
    iv[2] = uint(floor(p.z));

    vec3 result = hash3(iv);

    return result;
}

// https://docs.chaosgroup.com/display/OSLShaders/Flakes+normal+map

uniform float flake_scale = 50.0;               // Smaller values zoom into the flake map, larger values zoom out.
uniform float flake_size = 0.5;                 // Relative size of the flakes
uniform float flake_size_variance = 0.7;        // 0.0 makes all flakes the same size, 1.0 assigns random size between 0 and the given flake size
uniform float flake_normal_orientation = 0.5;   // Blend between the flake normals (0.0) and the surface normal (1.0)

void flakes(
    float u,
    float v,
    out vec3 result,
    out float alpha)
{
    float safe_flake_size_variance = clamp(flake_size_variance, 0.1, 1.0);

    vec3 cellCenters[9] = {
        vec3(0.5, 0.5, 0.0),
        vec3(1.5, 0.5, 0.0),
        vec3(1.5, 1.5, 0.0),
        vec3(0.5, 1.5, 0.0),
        vec3(-0.5, 1.5, 0.0),
        vec3(-0.5, 0.5, 0.0),
        vec3(-0.5, -0.5, 0.0),
        vec3(0.5, -0.5, 0.0),
        vec3(1.5, -0.5, 0.0)
    };

    vec3 position = vec3(u, v, 0.0);
    position = flake_scale * position;

    vec3 base = floor(position);

    vec3 nearestCell = vec3(0.0, 0.0, 1.0);
    int nearestCellIndex = -1;

    for (int cellIndex = 0; cellIndex < 9; ++cellIndex)   {
        vec3 cellCenter = base + cellCenters[cellIndex];

        vec3 centerOffset = cellnoise(cellCenter) * 2.0 - 1.0;
        centerOffset[2] *= safe_flake_size_variance;
        centerOffset = normalize(centerOffset);

        cellCenter += 0.5 * centerOffset;
        float cellDistance = distance(position, cellCenter);

        if (cellDistance < flake_size && cellCenter[2] < nearestCell[2]) {
            nearestCell = cellCenter;
            nearestCellIndex = cellIndex;
        }
    }

    result = vec3(0.5, 0.5, 1.0);
    alpha = 0.0;

    vec3 I = vec3(0, 0, 1);

    if (nearestCellIndex != -1) {
        vec3 randomNormal = cellnoise(base + cellCenters[nearestCellIndex] + vec3(0.0, 0.0, 1.5));
        randomNormal = 2.0 * randomNormal - 1.0;
        randomNormal = faceforward(randomNormal, I, randomNormal);
        randomNormal = normalize(mix(randomNormal, vec3(0.0, 0.0, 1.0), flake_normal_orientation));

        result = vec3(0.5*randomNormal[0] + 0.5, 0.5*randomNormal[1] + 0.5, randomNormal[2]);
        alpha = 1.0;
    }
}

void main()
{
    vec2 uv = gl_FragCoord.xy / u_resolution.xy;

    vec3 result;
    float alpha;

    flakes(
        uv.x, uv.y,
        result,
        alpha);

    gl_FragColor.xyz = result;
    gl_FragColor.w = alpha;
}
#endif