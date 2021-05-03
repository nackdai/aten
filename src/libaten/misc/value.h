#pragma once

#include <map>
#include <memory>
#include <string>
#include <variant>

#include "types.h"
#include "math/vec3.h"

template<typename T> struct is_shared_ptr : std::false_type {};
template<typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

namespace aten {
    class PolymorphicValue {
    public:
        PolymorphicValue() = default;
        ~PolymorphicValue() = default;

        PolymorphicValue(const PolymorphicValue& rhs)
        {
            val_ = rhs.val_;
            type_hash_ = rhs.type_hash_;
        }

        PolymorphicValue(PolymorphicValue&& rhs) noexcept
        {
            val_ = std::move(rhs.val_);
            type_hash_ = rhs.type_hash_;
        }
        PolymorphicValue& operator=(const PolymorphicValue& rhs) = delete;
        PolymorphicValue& operator=(PolymorphicValue&& rhs) = delete;

        PolymorphicValue& operator=(real f)
        {
            val_ = f;
            return *this;
        }
        PolymorphicValue& operator=(int i)
        {
            val_ = i;
            return *this;
        }
        PolymorphicValue& operator=(bool b)
        {
            val_ = b;
            return *this;
        }
        PolymorphicValue& operator=(const vec4& v)
        {
            val_ = v;
            return *this;
        }
        template <typename T>
        PolymorphicValue& operator=(const std::shared_ptr<T> _p)
        {
            type_hash_ = typeid(T).hash_code();
            val_ = _p;
            return *this;
        }

        operator real() const
        {
            return std::get<0>(val_);
        }
        operator int() const
        {
            return std::get<1>(val_);
        }
        operator bool() const
        {
            return std::get<2>(val_);
        }
        operator vec4() const
        {
            return std::get<3>(val_);
        }
        operator vec3() const
        {
            return static_cast<vec3>(std::get<3>(val_));
        }
        template <typename T>
        operator const std::shared_ptr<T>&() const
        {
            assert(type_hash_ > 0 && type_hash_ == typeid(T).hash_code());
            return std::reinterpret_pointer_cast<T>(std::get<4>(val_));
        }

        template <typename T>
        auto getAs() const -> std::enable_if_t<is_shared_ptr<T>::value, T>
        {
            assert(type_hash_ > 0 && type_hash_ == typeid(typename T::element_type).hash_code());
            return std::reinterpret_pointer_cast<typename T::element_type>(std::get<4>(val_));
        }

        template <typename T>
        auto getAs() const -> std::enable_if_t<!is_shared_ptr<T>::value, T>
        {
            //return static_cast<T>(*this);
            if constexpr (std::is_same_v<T, real>) {
                return this->operator real();
            }
            else if constexpr (std::is_same_v<T, bool>) {
                return this->operator bool();
            }
            else if constexpr (std::is_same_v<T, vec4>) {
                return this->operator vec4();
            }
            else if constexpr (std::is_same_v<T, vec3>) {
                return this->operator vec3();
            }
            else {
                return this->operator int();
            }
        }

    private:
        std::variant<real, int, bool, vec4, std::shared_ptr<void>> val_;
        std::size_t type_hash_{ 0 };
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
        TYPE get(std::string s, const TYPE& defaultValue) const
        {
            auto it = find(s);
            if (it != end()) {
                const PolymorphicValue& v = it->second;
                return v.getAs<TYPE>();
            }

            return defaultValue;
        }

        template <typename TYPE>
        std::shared_ptr<TYPE> get(std::string s) const
        {
            auto it = find(s);
            if (it != end()) {
                const PolymorphicValue& v = it->second;
                return v.getAs<std::shared_ptr<TYPE>>();
            }

            return std::shared_ptr<TYPE>();
        }
    };
}
