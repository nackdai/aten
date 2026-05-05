#pragma once

#include <array>

#include "defs.h"

namespace aten {
    enum class MeterUnit {
        m = 0,
        km,
        cm,
        mm,
        nm,
        max,
    };

    namespace _detail {
        AT_DEVICE_API constexpr std::array<float, static_cast<size_t>(MeterUnit::max)> LengthInMeters = {
            1.0F,    // m
            1000.0F, // km
            1e-2F,   // cm
            1e-3F,   // mm
            1e-9F,   // nm
        };

        AT_DEVICE_API constexpr float constexpr_pow(float base, int32_t exp) {
            if (exp == 0) {
                return 1.0F;
            }
            if (exp < 0) {
                return 1.0F / constexpr_pow(base, -exp);
            }
            float res = 1.0F;
            for (int32_t i = 0; i < exp; ++i) {
                res *= base;
            }
            return res;
        }

        template <int32_t Power>
        AT_DEVICE_API constexpr float get_conversion_factor(MeterUnit unit) {
            float unit_m = LengthInMeters[static_cast<size_t>(unit)];
            return constexpr_pow(unit_m, Power);
        }
    }

    template <int32_t TPower>
    class Quantity {
    public:
        static constexpr auto Power = TPower;

        AT_DEVICE_API constexpr explicit Quantity(float v) : value_{ v } {}

        AT_DEVICE_API constexpr float as(MeterUnit unit) const
        {
            float unit_m = _detail::LengthInMeters[static_cast<size_t>(unit)];
            return value_ / _detail::constexpr_pow(unit_m, Power);
        }

        AT_DEVICE_API constexpr operator float() const { return value_; }

        static AT_DEVICE_API constexpr float as(float v, MeterUnit unit)
        {
            float unit_m = _detail::LengthInMeters[static_cast<size_t>(unit)];
            return v / _detail::constexpr_pow(unit_m, Power);
        }

        static AT_DEVICE_API constexpr float from(float v, MeterUnit src, MeterUnit dst)
        {
            float unit_src = _detail::LengthInMeters[static_cast<size_t>(src)];
            return as(v * unit_src, dst);
        }

    private:
        float value_;
        static_assert(Power != 0, "Power cannot be zero");
    };

    template <int32_t Power>
    inline AT_DEVICE_API constexpr Quantity<Power> operator+(const Quantity<Power>& a, const Quantity<Power>& b)
    {
        return Quantity<Power>(static_cast<float>(a) + static_cast<float>(b));
    }

    template <int32_t Power>
    inline AT_DEVICE_API constexpr Quantity<Power> operator+(const Quantity<Power>& a, float f)
    {
        return Quantity<Power>(static_cast<float>(a) + f);
    }

    template <int32_t Power>
    inline AT_DEVICE_API constexpr Quantity<Power> operator-(const Quantity<Power>& a, const Quantity<Power>& b)
    {
        return Quantity<Power>(static_cast<float>(a) - static_cast<float>(b));
    }

    template <int32_t Power>
    inline AT_DEVICE_API constexpr Quantity<Power> operator-(const Quantity<Power>& a, float f)
    {
        return Quantity<Power>(static_cast<float>(a) - f);
    }

    template <int32_t Power>
    inline AT_DEVICE_API constexpr Quantity<Power> operator*(const Quantity<Power>& a, const Quantity<Power>& b)
    {
        return Quantity<Power>(static_cast<float>(a) * static_cast<float>(b));
    }

    template <int32_t Power>
    inline AT_DEVICE_API constexpr Quantity<Power> operator*(const Quantity<Power>& a, float f)
    {
        return Quantity<Power>(static_cast<float>(a) * f);
    }

    template <int32_t Power>
    inline AT_DEVICE_API constexpr Quantity<Power> operator/(const Quantity<Power>& a, const Quantity<Power>& b)
    {
        return Quantity<Power>(static_cast<float>(a) / static_cast<float>(b));
    }

    template <int32_t Power>
    inline AT_DEVICE_API constexpr Quantity<Power> operator/(const Quantity<Power>& a, float f)
    {
        return Quantity<Power>(static_cast<float>(a) / f);
    }

    using Length = aten::Quantity<1>;
    using InverseLength = aten::Quantity<-1>;
}

// User-defined literals.
// Provides support for the "10.0_km" literal syntax.
AT_DEVICE_API constexpr aten::Length operator"" _m(long double v) {
    return aten::Length(static_cast<float>(v));
}
AT_DEVICE_API constexpr aten::Length operator"" _km(long double v) {
    return aten::Length(static_cast<float>(v) * aten::_detail::get_conversion_factor<aten::Length::Power>(aten::MeterUnit::km));
}
AT_DEVICE_API constexpr aten::Length operator"" _cm(long double v) {
    return aten::Length(static_cast<float>(v) * aten::_detail::get_conversion_factor<aten::Length::Power>(aten::MeterUnit::cm));
}
AT_DEVICE_API constexpr aten::Length operator"" _mm(long double v) {
    return aten::Length(static_cast<float>(v) * aten::_detail::get_conversion_factor<aten::Length::Power>(aten::MeterUnit::mm));
}
AT_DEVICE_API constexpr aten::Length operator"" _nm(long double v) {
    return aten::Length(static_cast<float>(v) * aten::_detail::get_conversion_factor<aten::Length::Power>(aten::MeterUnit::nm));
}

AT_DEVICE_API constexpr aten::InverseLength operator"" _per_m(long double v) {
    return aten::InverseLength(static_cast<float>(v));
}
AT_DEVICE_API constexpr aten::InverseLength operator"" _per_km(long double v) {
    return aten::InverseLength(static_cast<float>(v) * aten::_detail::get_conversion_factor<aten::InverseLength::Power>(aten::MeterUnit::km));
}
AT_DEVICE_API constexpr aten::InverseLength operator"" _per_cm(long double v) {
    return aten::InverseLength(static_cast<float>(v) * aten::_detail::get_conversion_factor<aten::InverseLength::Power>(aten::MeterUnit::cm));
}
AT_DEVICE_API constexpr aten::InverseLength operator"" _per_mm(long double v) {
    return aten::InverseLength(static_cast<float>(v) * aten::_detail::get_conversion_factor<aten::InverseLength::Power>(aten::MeterUnit::mm));
}
AT_DEVICE_API constexpr aten::InverseLength operator"" _per_nm(long double v) {
    return aten::InverseLength(static_cast<float>(v) * aten::_detail::get_conversion_factor<aten::InverseLength::Power>(aten::MeterUnit::nm));
}
