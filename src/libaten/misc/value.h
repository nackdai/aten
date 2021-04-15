#pragma once

#include <map>
#include <memory>
#include <string>

#include "types.h"
#include "math/vec3.h"

namespace aten {
    class PolymorphicValue {
    public:
        union _value {
            real f;
            int i;
            bool b;
            vec4 v;

            _value() {}
            ~_value() {}
        };

        _value val;
        std::shared_ptr<void> p;

        PolymorphicValue()
        {
        }
        PolymorphicValue(const PolymorphicValue& rhs)
        {
            val = rhs.val;
            p = rhs.p;
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
        PolymorphicValue& operator=(const vec3& _v)
        {
            val.v = _v;
            return *this;
        }
        PolymorphicValue& operator=(const vec4& _v)
        {
            val.v = _v;
            return *this;
        }
        template <typename T>
        PolymorphicValue& operator=(T* _p)
        {
            p.reset();
            p = std::shared_ptr<void>(_p);
            return *this;
        }
        template <typename T>
        PolymorphicValue& operator=(std::shared_ptr<T>& _p)
        {
            p = _p;
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
        operator vec3() const
        {
            return val.v;
        }
        operator vec4() const
        {
            return val.v;
        }
        operator void*() const
        {
            return p.get();
        }

        template <typename TYPE>
        TYPE getAs() const
        {
            AT_ASSERT(false);
            return *(TYPE*)p.get();
        }

#if defined(_WIN32) || defined(_WIN64)
        template <>
        real PolymorphicValue::getAs() const;

        template <>
        int PolymorphicValue::getAs() const;

        template <>
        bool PolymorphicValue::getAs() const;

        template <>
        vec3 PolymorphicValue::getAs() const;

        template <>
        void* PolymorphicValue::getAs() const;
#endif
    };

    template <>
    inline real PolymorphicValue::getAs() const
    {
        return val.f;
    }
    template <>
    inline int PolymorphicValue::getAs() const
    {
        return val.i;
    }
    template <>
    inline bool PolymorphicValue::getAs() const
    {
        return val.b;
    }
    template <>
    inline vec3 PolymorphicValue::getAs() const
    {
        return val.v;
    }
    template <>
    inline vec4 PolymorphicValue::getAs() const
    {
        return val.v;
    }
    template <>
    inline void* PolymorphicValue::getAs() const
    {
        return p.get();
    }

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
