#pragma once

#include "defs.h"
#include "math/math.h"
#include "math/vec3.h"

namespace aten {
	class vec4 {
	public:
		union {
			vec3 v;
			struct {
				real x, y, z, w;
			};
			struct {
				real r, g, b, a;
			};
			real p[4];
		};

		vec4()
		{
			x = y = z = 0;
			w = 1;
		}
		vec4(const vec4& _v)
		{
			v = _v.v;
			w = _v.w;
		}
		vec4(real f)
		{
			x = y = z = w = f;
		}
		vec4(real _x, real _y, real _z, real _w)
		{
			x = _x;
			y = _y;
			z = _z;
			w = _w;
		}
		vec4(const vec3& _v, real _w)
		{
			v = _v;
			w = _w;
		}

		inline operator vec3() const
		{
			return std::move(v);
		}
		inline real operator[](int i) const
		{
			return p[i];
		}
		inline real& operator[](int i)
		{
			return p[i];
		}

		inline const vec4& set(real _x, real _y, real _z, real _w)
		{
			x = _x;
			y = _y;
			z = _z;
			w = _w;
			return *this;
		}
		inline const vec4& set(const vec3& _v, real _w)
		{
			v = _v;
			w = _w;
			return *this;
		}

		inline const vec4& operator+() const
		{
			return *this;
		}
		inline vec4 operator-() const
		{
			return vec4(-x, -y, -z, -w);
		}

		inline vec4& operator+=(const vec4& _v)
		{
			x += _v.x;
			y += _v.y;
			z += _v.z;
			w += _v.w;
			return *this;
		}
		inline vec4& operator-=(const vec4& _v)
		{
			x -= _v.x;
			y -= _v.y;
			z -= _v.z;
			w -= _v.w;
			return *this;
		}
		inline vec4& operator*=(const vec4& _v)
		{
			x *= _v.x;
			y *= _v.y;
			z *= _v.z;
			w *= _v.w;
			return *this;
		}
		inline vec4& operator/=(const vec4& _v)
		{
			x /= _v.x;
			y /= _v.y;
			z /= _v.z;
			w /= _v.w;
			return *this;
		}
		inline vec4& operator*=(const real t)
		{
			x *= t;
			y *= t;
			z *= t;
			w *= t;
			return *this;
		}
		inline vec4& operator/=(const real t)
		{
			x /= t;
			y /= t;
			z /= t;
			w /= t;
			return *this;
		}

		inline real length() const
		{
			auto ret = aten::sqrt(x * x + y * y + z * z + w * w);
			return ret;
		}
		inline real squared_length() const
		{
			auto ret = x * x + y * y + z * z + w * w;
			return ret;
		}

		void normalize()
		{
			auto l = length();
			*this /= l;
		}
	};

	inline vec4 operator+(const vec4& v1, const vec4& v2)
	{
		vec4 ret(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
		return std::move(ret);
	}

	inline vec4 operator+(const vec4 v1, real f)
	{
		vec4 ret(v1.x + f, v1.y + f, v1.z + f, v1.w + f);
		return std::move(ret);
	}

	inline vec4 operator-(const vec4& v1, const vec4& v2)
	{
		vec4 ret(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w);
		return std::move(ret);
	}

	inline vec4 operator-(const vec4& v1, real f)
	{
		vec4 ret(v1.x - f, v1.y - f, v1.z - f, v1.w - f);
		return std::move(ret);
	}

	inline vec4 operator*(const vec4& v1, const vec4& v2)
	{
		vec4 ret(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z, v1.w * v2.w);
		return std::move(ret);
	}

	inline vec4 operator/(const vec4& v1, const vec4& v2)
	{
		vec4 ret(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z, v1.w / v2.w);
		return std::move(ret);
	}

	inline vec4 operator*(real t, const vec4& v)
	{
		vec4 ret(t * v.x, t * v.y, t * v.z, t * v.w);
		return std::move(ret);
	}

	inline vec4 operator*(const vec4& v, real t)
	{
		vec4 ret(t * v.x, t * v.y, t * v.z, t * v.w);
		return std::move(ret);
	}

	inline vec4 operator/(const vec4& v, real t)
	{
		vec4 ret(v.x / t, v.y / t, v.z / t, v.w / t);
		return std::move(ret);
	}

	inline real dot(const vec4& v1, const vec4& v2)
	{
		auto ret = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
		return ret;
	}

	inline vec4 cross(const vec4& v1, const vec4& v2)
	{
		vec4 ret(
			v1.p[1] * v2.p[2] - v1.p[2] * v2.p[1],
			v1.p[2] * v2.p[0] - v1.p[0] * v2.p[2],
			v1.p[0] * v2.p[1] - v1.p[1] * v2.p[0],
			0);

		return std::move(ret);
	}

	inline vec4 normalize(const vec4& v)
	{
		auto ret = v / v.length();
		return std::move(ret);
	}
}
