#include "math/mat4.h"

namespace aten {
	const mat4 mat4::Identity(
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1);

	const mat4 mat4::Zero(
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0);

	inline void swap(vec4& r1, vec4& r2)
	{
		vec4 _t;
		_t = r1;

		r1 = r2;
		r2 = _t;
	}

	inline vec4 scale(const vec4& v, real f)
	{
		vec4 ret = v;
		ret[0] *= f;
		ret[1] *= f;
		ret[2] *= f;
		ret[3] *= f;

		return std::move(ret);
	}

	inline vec4 sub(const vec4& v1, const vec4& v2)
	{
		vec4 ret = v1;
		ret[0] -= v2[0];
		ret[1] -= v2[1];
		ret[2] -= v2[2];
		ret[3] -= v2[3];

		return std::move(ret);
	}

	mat4& mat4::invert()
	{
		// Gauss/Jordan法で求める
		mat4 mtx = *this;
		mat4 dst;

		for (int i = 0; i < 4; ++i) {
			// ピボット選択.
			// NOTE: 対象となる列中の最大値が対角値になるように行を入れ替える.
			real f = aten::abs(mtx.m[i][i]);
			for (int j = i + 1; j < 4; ++j) {
				if (f < aten::abs(mtx.m[j][i])) {
					f = aten::abs(mtx.m[j][i]);
					swap(mtx.v[i], mtx.v[j]);
					swap(dst.v[i], dst.v[j]);
				}
			}

			// 対象となる行の対角値を 1 にする.
			f = 1.0f / mtx.m[i][i];
			mtx.v[i] = scale(mtx.v[i], f);
			dst.v[i] = scale(dst.v[i], f);

			// 対象とならない列の値を 0 にする.
			for (int j = 0; j < 4; ++j) {
				if (j != i) {
					real temp = mtx.m[j][i];

					vec4 v1 = scale(mtx.v[i], temp);
					vec4 v2 = scale(dst.v[i], temp);

					mtx.v[j] = sub(mtx.v[j], v1);
					dst.v[j] = sub(dst.v[j], v2);
				}
			}
		}

		*this = dst;

		return *this;
	}

	mat4& mat4::asRotateByAxis(real r, const vec3& axis)
	{
		// NOTE
		// http://www.cg.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/tech07.html

		const real x = axis.x;
		const real y = axis.y;
		const real z = axis.z;

		const real c = aten::cos(r);
		const real s = aten::sin(r);

		m00 = x * x * (1 - c) + c;
		m01 = x * y * (1 - c) - z * s;
		m02 = z * x * (1 - c) + y * s;
		m03 = 0;

		m10 = x * y * (1 - c) + z * s;
		m11 = y * y * (1 - c) + c;
		m12 = y * z * (1 - c) - x * s;
		m13 = 0;

		m20 = z * x * (1 - c) - y * s;
		m21 = y * z * (1 - c) + x * s;
		m22 = z * z * (1 - c) + c;
		m23 = 0;

		m30 = 0; 
		m31 = 0;
		m32 = 0;
		m33 = 1;

		return *this;
	}
}
