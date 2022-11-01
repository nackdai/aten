#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec4 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec4 albedo;

uniform vec3 pointLitPos;
uniform vec3 pointLitClr;
uniform vec3 pointLitAttr;
uniform vec3 cameraPos;
uniform float ni = 1.0;
uniform float nt = 3.0;
uniform float roughness = 0.25;

uniform bool hasAlbedo = false;

layout(location = 0) out vec4 outColor;

#define AT_MATH_PI 3.141592653589793
#define gamma 2.2

#define Step (AT_MATH_PI / 2 / 40)
#define TableSize 25

float[TableSize][2] EffectiveRetroreflectiveAreaTable = float[][](
    float[](0.000000, 0.672),
    float[](0.039270, 0.669),
    float[](0.078540, 0.662),
    float[](0.117810, 0.651),
    float[](0.157080, 0.634),
    float[](0.196350, 0.612),
    float[](0.235619, 0.589),
    float[](0.274889, 0.559),
    float[](0.314159, 0.526),
    float[](0.353429, 0.484),
    float[](0.392699, 0.438),
    float[](0.431969, 0.389),
    float[](0.471239, 0.336),
    float[](0.510509, 0.281),
    float[](0.549779, 0.223),
    float[](0.589049, 0.161),
    float[](0.628319, 0.128),
    float[](0.667588, 0.109),
    float[](0.706858, 0.092),
    float[](0.746128, 0.072),
    float[](0.785398, 0.047),
    float[](0.824668, 0.034),
    float[](0.863938, 0.018),
    float[](0.903208, 0.008),
    float[](0.942478, 0.001)
);

float getEffectiveRetroreflectiveArea(vec3 into_prismatic_sheet_dir, vec3 normal)
{
    float c = dot(into_prismatic_sheet_dir, -normal);
    if (c < 0.0) {
        return 0.0;
    }

    float theta = acos(c);

    int idx = int(theta / Step);

    float a = 0.0;
    float b = 0.0;
    float t = 0.0;

    if (idx >= TableSize) {
        return 0.0;
    }
    else {
        float d = EffectiveRetroreflectiveAreaTable[idx][0];
        t = min(1.0, abs(d - theta) / Step);

        a = EffectiveRetroreflectiveAreaTable[idx][1];
        if (idx < TableSize - 1) {
            // Not end of the table.
            b = EffectiveRetroreflectiveAreaTable[idx + 1][1];
        }
    }

    float result = a * (1 - t) + b * t;
    return result;
}

float sampleBeckman_D(vec3 wh, vec3 n, float alpha)
{
    // NOTE
    // https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/

    float costheta = dot(wh, n);

    if (costheta <= 0) {
        return 0;
    }

    float cos2 = costheta * costheta;

    float sintheta = sqrt(1 - clamp(cos2, 0, 1));
    float tantheta = sintheta / costheta;
    float tan2 = tantheta * tantheta;

    float a2 = alpha * alpha;

    float D = 1 / (AT_MATH_PI * a2 * cos2 * cos2);
    D *= exp(-tan2 / a2);

    return D;
}

float sampleBeckman_G(vec3 n, vec3 v, vec3 m, float alpha)
{
    // http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html

    float vm = dot(v, m);
    float vn = dot(v, n);

    bool is_valid = (vn > 0) && (vm / vn > 0);

    if (is_valid) {
        float a = vn / (alpha * sqrt(1 - vn * vn));
        float a2 = a * a;

        if (a < 1.6) {
            return (3.535 * a + 2.181 * a2) / (1 + 2.276 * a + 2.577 * a2);
        }
        else {
            return 1;
        }
    }

    return 0;
}

float compute_F(vec3 L, vec3 H, float ni, float nt)
{
    float F = 1;
    {
        // http://d.hatena.ne.jp/hanecci/20130525/p3

        // NOTE
        // Fschlick(v,h) à R0 + (1 - R0)(1 - cosƒ¦)^5
        // R0 = ((n1 - n2) / (n1 + n2))^2

        float r0 = (ni - nt) / (ni + nt);
        r0 = r0 * r0;

        float LdotH = abs(dot(L, H));

        F = r0 + (1 - r0) * pow((1 - LdotH), 5);
    }

    return F;
}

vec3 computeBeckman(vec3 albedo, vec3 wi, vec3 wo, vec3 normal, float ni, float nt, float roughness)
{
    vec3 V = -wi;
    vec3 L = wo;
    vec3 N = normal;
    vec3 H = normalize(V + L);

    float alpha = roughness;

    float D = sampleBeckman_D(H, N, alpha);
    float G = sampleBeckman_G(V, N, H, alpha) * sampleBeckman_G(L, N, H, alpha);
    float F = compute_F(L, H, ni, nt);

    float NdotL = abs(dot(N, L));
    float NdotV = abs(dot(N, V));
    float denom = 4 * NdotL * NdotV;

    vec3 bsdf = denom > 0 ? vec3(F * G * D / denom) : vec3(0);
    return bsdf;
}

vec3 computeLambert(vec3 albedo)
{
    return albedo / AT_MATH_PI;
}

vec3 computeRetroreflective(vec3 albedo, vec3 wi, vec3 wo, vec3 normal, float ni, float nt, float roughness)
{
    vec3 V = -wi;
    vec3 L = wo;
    vec3 N = normalize(normal);
    vec3 H = normalize(V + L);

    vec3 V_dash = reflect(-V, N);
    vec3 B = normalize(L + V_dash);

    float D = sampleBeckman_D(B, N, roughness);
    float G = sampleBeckman_G(V, N, H, roughness) * sampleBeckman_G(L, N, H, roughness);
    //float F = compute_F(V_dash, B, ni, nt);
    float F = compute_F(V, B, ni, nt);

    float NdotL = abs(dot(N, L));
    float NdotV = abs(dot(N, V));
    float denom = 4 * NdotL * NdotV;

#if 1
    vec3 bsdf = denom > 0 ? vec3(F * G * D / denom) : vec3(0);
#else
    float costheta = dot(B, N);

    //L = L * 0.5 + 0.5;
    float cos2 = costheta * costheta;

    float angle = acos(costheta);
    float s = sin(angle);

    float sintheta = sqrt(1 - clamp(cos2, 0, 1));
    float tantheta = sintheta / costheta;
    float tan2 = tantheta * tantheta;
    float c = dot(B, H);
    vec3 bsdf = vec3(F);
#endif

    return bsdf;
}

void main()
{
    float d = length(pointLitPos - pos.xyz);
    float attn = 1.0 / (pointLitAttr.x + pointLitAttr.y * d + pointLitAttr.z * d * d);
    vec3 light = pointLitClr / attn;

    vec3 wi = normalize(pos.xyz - cameraPos);;
    vec3 wo = normalize(pointLitPos - pos.xyz);

    vec3 bsdfBeckman = computeBeckman(albedo.xyz, wi, wo, normal, ni, nt, roughness);
    vec3 bsdfLambert = computeLambert(albedo.xyz);
    vec3 bsdfRetroreflective = computeRetroreflective(albedo.xyz, wi, wo, normal, ni, nt, roughness);

    float costheta = dot(wo, normal);

    if (costheta < 0) {
        costheta = 0;
    }

    vec3 V = normalize(-wi);
    vec3 L = normalize(wo);
    vec3 N = normalize(normal);
    vec3 H = normalize(V + L);
    float F = compute_F(L, H, ni, nt);

    vec3 V_dash = reflect(-V, N);
    vec3 B = normalize(L + V_dash);
    float Fb = compute_F(V, B, ni, nt);

    vec3 r = refract(V, N, ni / nt);
    r = normalize(r);
    float era = getEffectiveRetroreflectiveArea(r, N);

    vec3 bsdf = albedo.xyz * (era * (bsdfBeckman + (1 - F) * bsdfRetroreflective) + (1-era) * bsdfLambert);

    outColor.xyz = bsdf * light * costheta;
    outColor.xyz = pow(outColor.xyz, vec3(1.0 / gamma));
    outColor.a = 1.0;
}

// NOTE
// https://developpaper.com/pbr-brdf-disney-unity-1/
// https://gist.github.com/D4KU/dc9467e2f77bdb2069dac964f2d1e7ec
// https://github.com/synthesis-labs/polaris-streams-3d/blob/master/unity-client/Library/PackageCache/com.unity.render-pipelines.lightweight%405.7.2/ShaderLibrary/Lighting.hlsl
