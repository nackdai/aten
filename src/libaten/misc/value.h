#pragma once

#include <string>
#include <map>
#include "types.h"
#include "math/vec3.h"

namespace aten {
	class PolymorphicValue {
	public:
		union _value {
			real f;
			int i;
			bool b;
			std::string* s;
			vec3 v;
			void* p;

			_value() {}
			~_value() {}
		};

		_value val;

		PolymorphicValue()
		{
			val.s = new std::string();
		}
		PolymorphicValue(const PolymorphicValue& rhs)
		{
			val.s = new std::string();
			memcpy(&val, &rhs.val, sizeof(_value));
		}
		~PolymorphicValue() {}

		PolymorphicValue& operator=(real _f)
		{
			val.f = _f;
			return *this;
		}
		PolymorphicValue& operator=(int _i)
		{
			val.i = _i;
			return *this;
		}
		PolymorphicValue& operator=(bool _b)
		{
			val.b = _b;
			return *this;
		}
		PolymorphicValue& operator=(const char* _ch)
		{
			*val.s = _ch;
			return *this;
		}
		PolymorphicValue& operator=(const std::string& _s)
		{
			*val.s = _s;
			return *this;
		}
		PolymorphicValue& operator=(const vec3& _v)
		{
			val.v = _v;
			return *this;
		}
		PolymorphicValue& operator=(void* _p)
		{
			val.p = _p;
			return *this;
		}

		operator real() const
		{
			return val.f;
		}
		operator int() const
		{
			return val.i;
		}
		operator bool() const
		{
			return val.b;
		}
		operator std::string() const
		{
			return *val.s;
		}
		operator const char*() const
		{
			return val.s->c_str();
		}
		operator vec3() const
		{
			return val.v;
		}
		operator void*() const
		{
			return val.p;
		}

		template <typename TYPE>
		TYPE getAs() const
		{
			AT_ASSERT(false);
			return *(TYPE*)val.p;
		}
		template <>
		real getAs()const
		{
			return val.f;
		}
		template <>
		int getAs() const
		{
			return val.i;
		}
		template <>
		bool getAs() const
		{
			return val.b;
		}
		template <>
		std::string getAs() const
		{
			return *val.s;
		}
		template <>
		const char* getAs() const
		{
			return val.s->c_str();
		}
		template <>
		vec3 getAs() const
		{
			return val.v;
		}
		template <>
		void* getAs() const
		{
			return val.p;
		}
	};

	class Values : public std::map<std::string, PolymorphicValue> {
	public:
		Values() {}
		~Values() {}

		void add(std::string& name, const PolymorphicValue& val)
		{
			this->insert(std::pair<std::string, PolymorphicValue>(name, val));
		}

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