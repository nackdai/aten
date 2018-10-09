#include <functional>

#include "deformable/DeformAnimationInterp.h"
#include "math/mat4.h"
#include "math/quaternion.h"

namespace aten
{
    bool DeformAnimationInterp::isScalarInterp(AnmInterpType type)
    {
        type = (AnmInterpType)((uint32_t)type & (uint32_t)AnmInterpType::Mask);

        bool ret = (type == AnmInterpType::Linear)
            || (type == AnmInterpType::Bezier)
            || (type == AnmInterpType::Hermite);

        return ret;
    }

    float DeformAnimationInterp::computeInterp(
        AnmInterpType nInterp,
        float fTime,
        uint32_t nKeyNum,
        uint32_t nPos,
        const AnmKey* pKeys)
    {
        using FuncInterpScalar = std::function<float(float, uint32_t, uint32_t, const AnmKey*)>;

        static FuncInterpScalar tblFuncInterp[] = {
            &DeformAnimationInterp::computeLinear,
            &DeformAnimationInterp::computeBezier,
            &DeformAnimationInterp::computeHermite,
            nullptr,
        };
        AT_STATICASSERT(AT_COUNTOF(tblFuncInterp) == (int32_t)AnmInterpType::Num);

        AT_ASSERT(tblFuncInterp[(int32_t)nInterp] != nullptr);

        float ret = (tblFuncInterp[(int32_t)nInterp])(fTime, nKeyNum, nPos, pKeys);
        return ret;
    }

    void DeformAnimationInterp::computeInterp(
        vec4& vRef,
        AnmInterpType nInterp,
        float fTime,
        uint32_t nKeyNum,
        uint32_t nPos,
        const AnmKey* pKeys)
    {
        DeformAnimationInterp::computeSlerp(vRef, fTime, nKeyNum, nPos, pKeys);
    }

    float DeformAnimationInterp::computeNomralizedTime(
        float fTime,
        int32_t& nPrev,
        int32_t& nNext,
        uint32_t nKeyNum,
        const AnmKey* pKeys)
    {
        nPrev = 0;
        nNext = -1;

        if (nKeyNum == 1)
        {
            return 0.0f;
        }

        float fPrevTime = pKeys[0].keyTime;

        for (uint32_t i = 1; i < nKeyNum; ++i) {
            float fNextTime = pKeys[i].keyTime;
            if ((fPrevTime <= fTime) && (fTime <= fNextTime)) {
                nPrev = i - 1;
                nNext = i;
                break;
            }
            fPrevTime = fNextTime;
        }

        if (nNext <= nPrev) {
            nNext = nKeyNum - 1;
            nPrev = nNext - 1;
        }

        AT_ASSERT(nNext > nPrev);

        // Normalize time 0 to 1.
        const float fStartTime = pKeys[nPrev].keyTime;
        const float fEndTime = pKeys[nNext].keyTime;
        AT_ASSERT(fStartTime < fEndTime);

        float fNormTime = (fTime - fStartTime) / (fEndTime - fStartTime);
        fNormTime = aten::clamp(fNormTime, 0.0f, 1.0f);

        return fNormTime;
    }

    float DeformAnimationInterp::computeLinear(
        float fTime,
        uint32_t nKeyNum,
        uint32_t nPos,
        const AnmKey* pKeys)
    {
        int32_t nPrev = 0;
        int32_t nNext = -1;

        float fNormTime = computeNomralizedTime(
            fTime,
            nPrev, nNext,
            nKeyNum,
            pKeys);

        AT_ASSERT(nPos < pKeys[nPrev].numParams);
        float param_0 = pKeys[nPrev].params[nPos];

        AT_ASSERT(nPos < pKeys[nNext].numParams);
        float param_1 = pKeys[nNext].params[nPos];

        float ret = param_0 * (1.0f - fNormTime) + param_1 * fNormTime;

        return ret;
    }

    float DeformAnimationInterp::computeBezier(
        float fTime,
        uint32_t nKeyNum,
        uint32_t nPos,
        const AnmKey* pKeys)
    {
        // TODO
        AT_ASSERT(false);
        return 0.0f;
    }

    float DeformAnimationInterp::computeHermite(
        float fTime,
        uint32_t nKeyNum,
        uint32_t nPos,
        const AnmKey* pKeys)
    {
        AT_ASSERT(pKeys != nullptr);

        float ret = 0.0f;

        const uint32_t KEY_PARAM_VALUE = nPos * 3;
        const uint32_t KEY_PARAM_IN_TANGENT = KEY_PARAM_VALUE + 1;
        const uint32_t KEY_PARAM_OUT_TANGENT = KEY_PARAM_IN_TANGENT + 1;

        // NOTE
        // s = (time - time0) / (time1 - time0) : Normalize time 0 to 1
        // S = { s^3, s^2, s^1, 1}
        // C = { P1, P2, T1, T2 }
        // b : Bezier matrix
        // P = S * b * c

        static const mat4 mtxBezier = {
#if 0
             2.0f, -2.0f,  1.0f,  1.0f,
            -3.0f,  3.0f, -2.0f, -1.0f,
             0.0f,  0.0f,  1.0f,  0.0f,
             1.0f,  0.0f,  0.0f,  0.0f,
#else
             2.0f, -3.0f, 0.0f, 1.0f,
            -2.0f,  3.0f, 0.0f, 0.0f,
             1.0f, -2.0f, 1.0f, 0.0f,
             1.0f, -1.0f, 0.0f, 0.0f,
#endif
        };

        if (pKeys[0].keyTime >= fTime) {
            ret = pKeys[0].params[KEY_PARAM_VALUE];
        }
        else if (pKeys[nKeyNum - 1].keyTime <= fTime) {
            ret = pKeys[nKeyNum - 1].params[KEY_PARAM_VALUE];
        }
        else {
            int32_t nPrev = 0;
            int32_t nNext = -1;

            float fNormTime = computeNomralizedTime(
                fTime,
                nPrev, nNext,
                nKeyNum,
                pKeys);

            float fNormTime_2 = fNormTime * fNormTime;

            vec4 vecS = {
                fNormTime_2 * fNormTime,
                fNormTime_2,
                fNormTime,
                1.0f,
            };

            vec4 vecC = {
                pKeys[nPrev].params[KEY_PARAM_VALUE],
                pKeys[nNext].params[KEY_PARAM_VALUE],
                pKeys[nPrev].params[KEY_PARAM_OUT_TANGENT],
                pKeys[nNext].params[KEY_PARAM_IN_TANGENT],
            };

            vecS = mtxBezier.apply(vecS);
            ret = dot(vecS, vecC);
        }

        return ret;
    }

    void DeformAnimationInterp::computeSlerp(
        vec4& vRef,
        float fTime,
        uint32_t nKeyNum,
        uint32_t nPos,
        const AnmKey* pKeys)
    {
        AT_ASSERT(pKeys != nullptr);

        if (pKeys[0].keyTime >= fTime) {
            vRef.x = pKeys[0].params[0];
            vRef.y = pKeys[0].params[1];
            vRef.z = pKeys[0].params[2];
            vRef.w = pKeys[0].params[3];
        }
        else if (pKeys[nKeyNum - 1].keyTime <= fTime) {
            vRef.x = pKeys[nKeyNum - 1].params[0];
            vRef.y = pKeys[nKeyNum - 1].params[1];
            vRef.z = pKeys[nKeyNum - 1].params[2];
            vRef.w = pKeys[nKeyNum - 1].params[3];
        }
        else {
            int32_t nPrev = 0;
            int32_t nNext = -1;

            float fNormTime = computeNomralizedTime(
                fTime,
                nPrev, nNext,
                nKeyNum,
                pKeys);

            quat quat1(
                pKeys[nPrev].params[0],
                pKeys[nPrev].params[1],
                pKeys[nPrev].params[2],
                pKeys[nPrev].params[3]);

            quat quat2(
                pKeys[nNext].params[0],
                pKeys[nNext].params[1],
                pKeys[nNext].params[2],
                pKeys[nNext].params[3]);

            // Slerp
            auto q = quat::slerp(quat1, quat2, fNormTime);
            vRef = vec4(q.x, q.y, q.z, q.w);
        }
    }
}