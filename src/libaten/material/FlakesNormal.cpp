#include "material/FlakesNormal.h"

namespace AT_NAME
{
    static inline AT_DEVICE_MTRL_API float bits_to_01(uint32_t bits)
    {
        // divide by 2^32-1
        uint32_t div = 0xffffffff;
        return bits * (1.0f / float(div));
    }

    static inline AT_DEVICE_MTRL_API uint32_t rotl32(uint32_t var, uint32_t hops)
    {
        return (var << hops) | (var >> (32 - hops));
    }

    // Bob Jenkins "lookup3" hashes:  http://burtleburtle.net/bob/c/lookup3.c
    // It's in the public domain.

    // Mix up the bits of a, b, and c (changing their values in place).
    static inline AT_DEVICE_MTRL_API void bjmix(uint32_t& a, uint32_t& b, uint32_t& c)
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
    static inline AT_DEVICE_MTRL_API uint32_t bjfinal(uint32_t a, uint32_t b, uint32_t c)
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

    static inline AT_DEVICE_MTRL_API uint32_t inthash(aten::vec4& k)
    {
        int N = 4;

        // now hash the data!
        uint32_t len = N;
        uint32_t a = 0xdeadbeef + (len << 2) + 13;
        uint32_t b = 0xdeadbeef + (len << 2) + 13;
        uint32_t c = 0xdeadbeef + (len << 2) + 13;

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
        a += (uint32_t)k[0];
        b += (uint32_t)k[1];
        c += (uint32_t)k[2];
        bjmix(a, b, c);

        a += (uint32_t)k[3];
        c = bjfinal(a, b, c);
#endif

        return c;
    }

    static inline AT_DEVICE_MTRL_API aten::vec3 hash3(aten::vec4& k)
    {
        int N = 4;

        aten::vec3 result;

        k[N - 1] = 0;
        result.x = bits_to_01(inthash(k));

        k[N - 1] = 1;
        result.y = bits_to_01(inthash(k));

        k[N - 1] = 2;
        result.z = bits_to_01(inthash(k));

        return result;
    }

    static inline AT_DEVICE_MTRL_API aten::vec3 cellnoise(const aten::vec3& p)
    {
        aten::vec4 iv;
        iv[0] = aten::floor(p.x);
        iv[1] = aten::floor(p.y);
        iv[2] = aten::floor(p.z);

        aten::vec3 result = hash3(iv);

        return result;
    }

    AT_DEVICE_MTRL_API aten::vec4 FlakesNormal::gen(
        real u, real v,
        real flake_scale/*= real(50.0)*/,
        real flake_size/*= real(0.5)*/,
        real flake_size_variance/*= real(0.7)*/,
        real flake_normal_orientation/*= real(0.5)*/)
    {
        float safe_flake_size_variance = aten::clamp(flake_size_variance, real(0.1), real(1.0));

        const aten::vec3 cellCenters[9] = {
            aten::vec3(0.5, 0.5, 0.0),
            aten::vec3(1.5, 0.5, 0.0),
            aten::vec3(1.5, 1.5, 0.0),
            aten::vec3(0.5, 1.5, 0.0),
            aten::vec3(-0.5, 1.5, 0.0),
            aten::vec3(-0.5, 0.5, 0.0),
            aten::vec3(-0.5, -0.5, 0.0),
            aten::vec3(0.5, -0.5, 0.0),
            aten::vec3(1.5, -0.5, 0.0)
        };

        aten::vec3 position(u, v, 0.0);
        position = flake_scale * position;

        aten::vec3 base = floor(position);

        aten::vec3 nearestCell(0.0, 0.0, 1.0);
        int nearestCellIndex = -1;

        for (int cellIndex = 0; cellIndex < 9; ++cellIndex) {
            aten::vec3 cellCenter = base + cellCenters[cellIndex];

            aten::vec3 centerOffset = cellnoise(cellCenter) * real(2.0) - real(1.0);
            centerOffset[2] *= safe_flake_size_variance;
            centerOffset = normalize(centerOffset);

            cellCenter += real(0.5) * centerOffset;
            float cellDistance = distance(position, cellCenter);

            if (cellDistance < flake_size && cellCenter[2] < nearestCell[2]) {
                nearestCell = cellCenter;
                nearestCellIndex = cellIndex;
            }
        }

        auto result = aten::vec3(0.5, 0.5, 1.0);
        real alpha = 0.0;

        aten::vec3 I(0, 0, 1);

        if (nearestCellIndex != -1) {

            aten::vec3 randomNormal = cellnoise(base + cellCenters[nearestCellIndex] + aten::vec3(0.0, 0.0, 1.5));
            randomNormal = real(2.0) * randomNormal - real(1.0);
            randomNormal = faceforward(randomNormal, I, randomNormal);
            randomNormal = normalize(aten::mix(randomNormal, aten::vec3(0.0, 0.0, 1.0), flake_normal_orientation));

            //result = aten::vec3(0.5*randomNormal[0] + 0.5, 0.5*randomNormal[1] + 0.5, randomNormal[2]);
            result = aten::vec3(randomNormal[0], randomNormal[1], randomNormal[2]);
            alpha = 1.0;
        }

        aten::vec4 ret = aten::vec4(result, alpha);

        return ret;
    }
}
