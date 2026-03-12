#include <gtest/gtest.h>

#include "atmosphere/sky/unit_quantity.h"

TEST(UnitQuantityTest, GetDirectValue)
{
    constexpr aten::Length m = 2.0_m;
    float f = static_cast<float>(m);
    EXPECT_EQ(f, 2.0F);
}

TEST(UnitQuantityTest, LengthAs)
{
    constexpr aten::Length km = 2.0_km;

    float f = km.as(aten::MeterUnit::m);
    EXPECT_EQ(f, 2000.0F);

    f = km.as(aten::MeterUnit::km);
    EXPECT_EQ(f, 2.0F);

    f = km.as(aten::MeterUnit::cm);
    EXPECT_EQ(f, 200000.0F);

    constexpr aten::Length m = 2.0_m;

    f = m.as(aten::MeterUnit::mm);
    EXPECT_NEAR(2e+3F, f, 1e-3F);

    f = m.as(aten::MeterUnit::nm);
    EXPECT_NEAR(2e+9F, f, 1e-1F);
}

TEST(UnitQuantityTest, InverseLengthAs)
{
    constexpr aten::InverseLength per_km = 2.0_per_km;

    float f = per_km.as(aten::MeterUnit::m);
    EXPECT_EQ(f, 2e-3F);

    f = per_km.as(aten::MeterUnit::km);
    EXPECT_EQ(f, 2.0F);

    constexpr aten::InverseLength per_m = 2.0_per_m;

    f = per_m.as(aten::MeterUnit::cm);
    EXPECT_EQ(f, 2e-2F);

    f = per_m.as(aten::MeterUnit::mm);
    EXPECT_EQ(f, 2e-3F);

    f = per_m.as(aten::MeterUnit::nm);
    EXPECT_EQ(f, 2e-9F);
}
