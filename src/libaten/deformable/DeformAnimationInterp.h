#pragma once

#include "ANMFormat.h"
#include "math/vec4.h"

namespace aten
{
    /**
     */
    class DeformAnimationInterp {
    private:
		DeformAnimationInterp();
        ~DeformAnimationInterp();

    public:
		static bool isScalarInterp(AnmInterpType type);

        static float computeInterp(
            AnmInterpType nInterp,
            float fTime,
            uint32_t nKeyNum,
            uint32_t nPos,
            const AnmKey* pKeys);

        static void computeInterp(
            vec4& vRef,
            AnmInterpType nInterp,
            float fTime,
            uint32_t nKeyNum,
            uint32_t nPos,
            const AnmKey* pKeys);

    private:
        static float computeLinear(
            float fTime,
            uint32_t nKeyNum,
            uint32_t nPos,
            const AnmKey* pKeys);

        static float computeBezier(
            float fTime,
            uint32_t nKeyNum,
            uint32_t nPos,
            const AnmKey* pKeys);

        static float computeHermite(
            float fTime,
            uint32_t nKeyNum,
            uint32_t nPos,
            const AnmKey* pKeys);

        static void computeSlerp(
            vec4& vRef,
            float fTime,
            uint32_t nKeyNum,
            uint32_t nPos,
            const AnmKey* pKeys);

        static void computeBezierSlerp(
            vec4& vRef,
            float fTime,
            uint32_t nKeyNum,
            uint32_t nPos,
            const AnmKey* pKeys);

    public:
        static float computeNomralizedTime(
            float fTime,
            int32_t& nPrev,
            int32_t& nNext,
            uint32_t nKeyNum,
            const AnmKey* pKeys);
    };
}
