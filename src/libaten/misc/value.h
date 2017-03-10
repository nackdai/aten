#pragma once

#include <string>
#include <map>
#include "types.h"
#include "math/vec3.h"

namespace aten {
	class PolymorphicValue {
	public:
		union {
			real f;
			int i;
			bool b;
			std::string s;
			vec3 v;
			void* p;
		};

		PolymorphicValue() {}
		PolymorphicValue(const PolymorphicValue& rhs)
		{
			
		}
		~PolymorphicValue() {}

		PolymorphicValue& operator=(real _f)
		{
			f = _f;
			return *this;
		}
		PolymorphicValue& operator=(int _i)
		{
			i = _i;
			return *this;
		}
		PolymorphicValue& operator=(bool _b)
		{
			b = _b;
			return *this;
		}
		PolymorphicValue& operator=(const char* _ch)
		{
			s = _ch;
			return *this;
		}
		PolymorphicValue& operator=(const std::string& _s)
		{
			s = _s;
			return *this;
		}
		PolymorphicValue& operator=(const vec3& _v)
		{
			v = _v;
			return *this;
		}
		PolymorphicValue& operator=(void* _p)
		{
			p = _p;
			return *this;
		}

		operator real() const
		{
			return f;
		}
		operator int() const
		{
			return i;
		}
		operator bool() const
		{
			return b;
		}
		operator std::string() const
		{
			return s;
		}
		operator const char*() const
		{
			return s.c_str();
		}
		operator vec3() const
		{
			return v;
		}
		operator void*() const
		{
			return p;
		}

		template <typename TYPE>
		TYPE getAs() const
		{
			AT_ASSERT(false);
			return *(TYPE*)p;
		}
		template <>
		real getAs()const
		{
			return f;
		}
		template <>
		int getAs() const
		{
			return i;
		}
		template <>
		bool getAs() const
		{
			return b;
		}
		template <>
		std::string getAs() const
		{
			return s;
		}
		template <>
		const char* getAs() const
		{
			return s.c_str();
		}
		template <>
		vec3 getAs() const
		{
			return v;
		}
		template <>
		void* getAs() const
		{
			return p;
		}
	};

	class Values : public std::map<std::string, PolymorphicValue> {
	public:
		Values() {}
		~Values() {}

		template <typename TYPE>
		TYPE get(std::string s, const TYPE& defaultValue)
		{
			auto it = find(s);
			if (it != end()) {
				PolymorphicValue v = it->second;
				TYPE ret = v.getAs<TYPE>();
				return ret;
			}

			return defaultValue;
		}
	};
}