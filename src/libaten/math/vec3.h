#pragma once

#include "defs.h"
#include "math/math.h"

namespace aten {
	class vec3 {
	public:
		union {
			struct {
				real x, y, z;
			};
			struct {
				real r, g, b;
			};
			real a[3];
		};

		vec3()
		{
			x = y = z = real(0);
		}

		vec3(real _x, real _y, real _z)
		{
			x = _x;
			y = _y;
			z = _z;
		}

		vec3(real f)
		{
			x = y = z = f;
		}

		inline AT_DEVICE_API const vec3& operator+() const
		{
			return *this;
		}
		inline AT_DEVICE_API vec3 operator-() const
		{
			vec3 ret(0);
			ret.x = -x;
			ret.y = -y;
			ret.z = -z;
			return ret;
		}
		inline AT_DEVICE_API real operator[](int i) const
		{
			return a[i];
		}
		inline AT_DEVICE_API real& operator[](int i)
		{
			return a[i];
		};

		inline AT_DEVICE_API vec3& operator+=(const vec3& v)
		{
			x += v.x;
			y += v.y;
			z += v.z;
			return *this;
		}
		inline AT_DEVICE_API vec3& operator-=(const vec3& v)
		{
			x -= v.x;
			y -= v.y;
			z -= v.z;
			return *this;
		}
		inline AT_DEVICE_API vec3& operator*=(const vec3& v)
		{
			x *= v.x;
			y *= v.y;
			z *= v.z;
			return *this;
		}
		inline AT_DEVICE_API vec3& operator/=(const vec3& v)
		{
			x /= v.x;
			y /= v.y;
			z /= v.z;
			return *this;
		}
		inline AT_DEVICE_API vec3& operator*=(const real t)
		{
			x *= t;
			y *= t;
			z *= t;
			return *this;
		}
		inline AT_DEVICE_API vec3& operator/=(const real t)
		{
			x /= t;
			y /= t;
			z /= t;
			return *this;
		}

		inline AT_DEVICE_API real length() const
		{
			auto ret = aten::sqrt(x * x + y * y + z * z);
			return ret;
		}
	};

	inline AT_DEVICE_API vec3 operator+(const vec3& v1, const vec3& v2)
	{
		vec3 ret = aten::vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
		return std::move(ret);
	}

	inline AT_DEVICE_API vec3 operator+(const vec3& v1, real f)
	{
		vec3 ret = aten::vec3(v1.x + f, v1.y + f, v1.z + f);
		return std::move(ret);
	}

	inline AT_DEVICE_API vec3 operator-(const vec3& v1, const vec3& v2)
	{
		vec3 ret = aten::vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
		return std::move(ret);
	}

	inline AT_DEVICE_API vec3 operator-(const vec3& v1, real f)
	{
		vec3 ret = aten::vec3(v1.x - f, v1.y - f, v1.z - f);
		return std::move(ret);
	}

	inline AT_DEVICE_API vec3 operator*(const vec3& v1, const vec3& v2)
	{
		vec3 ret = aten::vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
		return std::move(ret);
	}

	inline AT_DEVICE_API vec3 operator/(const vec3& v1, const vec3& v2)
	{
		vec3 ret = aten::vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
		return std::move(ret);
	}

	inline AT_DEVICE_API vec3 operator*(real t, const vec3& v)
	{
		vec3 ret = aten::vec3(t * v.x, t * v.y, t * v.z);
		return std::move(ret);
	}

	inline AT_DEVICE_API vec3 operator*(const vec3& v, real t)
	{
		vec3 ret = aten::vec3(t * v.x, t * v.y, t * v.z);
		return std::move(ret);
	}

	inline AT_DEVICE_API vec3 operator/(const vec3& v, real t)
	{
		vec3 ret = aten::vec3(v.x / t, v.y / t, v.z / t);
		return std::move(ret);
	}

	inline AT_DEVICE_API vec3 operator/(real t, const vec3& v)
	{
		vec3 ret = aten::vec3(t / v.x, t / v.y, t / v.z);
		return std::move(ret);
	}

	inline AT_DEVICE_API real dot(const vec3& v1, const vec3& v2)
	{
		auto ret = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
		return ret;
	}

	inline AT_DEVICE_API vec3 cross(const vec3& v1, const vec3& v2)
	{
		vec3 ret = aten::vec3(
			v1.a[1] * v2.a[2] - v1.a[2] * v2.a[1],
			v1.a[2] * v2.a[0] - v1.a[0] * v2.a[2],
			v1.a[0] * v2.a[1] - v1.a[1] * v2.a[0]);

		return std::move(ret);
	}

	inline AT_DEVICE_API vec3 normalize(const vec3& v)
	{
		auto invLen = aten::rsqrt(dot(v, v));
		auto ret = v * invLen;
		return std::move(ret);
	}

	inline AT_DEVICE_API real squared_length(const vec3& v)
	{
		auto ret = dot(v, v);
		return ret;
	}

	// 直行ベクトルを計算.
	inline AT_DEVICE_API vec3 getOrthoVector(const vec3& n)
	{
		vec3 p;

		// NOTE
		// dotを計算したときにゼロになるようなベクトル.
		// k は normalize 計算用.

#if 0
		if (aten::abs(n.z) > real(0.707106781186547524401)) {
			real k = aten::sqrt(n.y * n.y + n.z * n.z);
			p.x = 0; 
			p.y = -n.z / k; 
			p.z = n.y / k;
		}
		else {
			real k = aten::sqrt(n.x * n.x + n.y * n.y);
			p.x = -n.y / k; 
			p.y = n.x / k; 
			p.z = 0;
		}
#else
		if (aten::abs(n.z) > real(0)) {
			real k = aten::sqrt(n.y * n.y + n.z * n.z);
			p.x = 0;
			p.y = -n.z / k;
			p.z = n.y / k;
		}
		else {
			real k = aten::sqrt(n.x * n.x + n.y * n.y);
			p.x = n.y / k;
			p.y = -n.x / k;
			p.z = 0;
		}
#endif

		return std::move(p);
	}

	inline bool isInvalid(const vec3& v)
	{
		bool b0 = isInvalid(v.x);
		bool b1 = isInvalid(v.y);
		bool b2 = isInvalid(v.z);

		return b0 || b1 || b2;
	}
}
