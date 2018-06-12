#pragma once

#include "ANMFormat.h"
#include "math/vec4.h"

namespace aten
{
    /**
	 * @brief Interpolator to interpolate key frame data for deform animation.
     */
    class DeformAnimationInterp {
    private:
		DeformAnimationInterp();
        ~DeformAnimationInterp();

    public:
		/**
		 * @brief Return whether the specified type uses the interpolator with scalar.
		 */
		static bool isScalarInterp(AnmInterpType type);

		/**
		 * @brief Compute interpolated data between two key data.
		 */
        static float computeInterp(
            AnmInterpType nInterp,
            float fTime,
            uint32_t nKeyNum,
            uint32_t nPos,
            const AnmKey* pKeys);

		/**
		 * @brief Compute interpolated data between two key data.
		 */
        static void computeInterp(
            vec4& vRef,
            AnmInterpType nInterp,
            float fTime,
            uint32_t nKeyNum,
            uint32_t nPos,
            const AnmKey* pKeys);

    private:
		/**
		 * @brief Linear interpolator.
		 */
        static float computeLinear(
            float fTime,
            uint32_t nKeyNum,
            uint32_t nPos,
            const AnmKey* pKeys);

		/**
		 * @brief Bezier interpolator.
		 */
        static float computeBezier(
            float fTime,
            uint32_t nKeyNum,
            uint32_t nPos,
            const AnmKey* pKeys);

		/**
		 * @brief Hermite interpolator.
		 */
        static float computeHermite(
            float fTime,
            uint32_t nKeyNum,
            uint32_t nPos,
            const AnmKey* pKeys);

		/**
		 * @brief Spherical linear interpolator.
		 */
        static void computeSlerp(
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
