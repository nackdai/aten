#include <gtest/gtest.h>

#include <array>
#include <numeric>
#include <type_traits>
#include <vector>

#include "aten.h"

TEST(span_test, SpanConstructor)
{
    aten::span<int32_t> s0;
    ASSERT_EQ(s0.data(), nullptr);
    ASSERT_EQ(s0.size(), 0);

    const int32_t x = 0;
    aten::span s1(&x, 1);
    ASSERT_EQ(s1.data(), &x);
    ASSERT_EQ(s1.size(), 1);

    const int32_t a[4] = { 0, 1, 2, 3 };
    using c_style_array_type = std::remove_reference_t<decltype(std::declval<decltype(a)>()[0])>;
    aten::span s2(a);
    ASSERT_EQ(s2.data(), a);
    ASSERT_EQ(s2.size(), std::size(a));

    std::array<int32_t, 4> aa{ 0, 1, 2, 3 };
    aten::span s3(aa);
    ASSERT_EQ(s3.data(), aa.data());
    ASSERT_EQ(s3.size(), aa.size());

    std::vector<int32_t> v{ 0, 1, 2, 3 };
    aten::span s4(v);
    ASSERT_EQ(s4.data(), v.data());
    ASSERT_EQ(s4.size(), v.size());

    auto s5(s4);
    ASSERT_EQ(s5.data(), v.data());
    ASSERT_EQ(s5.size(), v.size());

    auto s6 = s5;
    ASSERT_EQ(s6.data(), v.data());
    ASSERT_EQ(s6.size(), v.size());
}

TEST(span_test, SpanSubspane)
{
    std::array<int32_t, 10> a;
    std::iota(a.begin(), a.end(), 0);

    aten::span<decltype(a)::value_type> s(a);
    ASSERT_EQ(s.data(), a.data());
    ASSERT_EQ(s.size(), a.size());

    auto first = s.first(5);
    ASSERT_EQ(first.data(), a.data());
    ASSERT_EQ(first.size(), 5);

    first = s.first(100);
    ASSERT_EQ(first.data(), nullptr);
    ASSERT_EQ(first.size(), 0);

    auto last = s.last(4);
    ASSERT_EQ(last.data(), a.data() + (a.size() - 4));
    ASSERT_EQ(last.size(), 4);

    last = s.last(100);
    ASSERT_EQ(last.data(), nullptr);
    ASSERT_EQ(last.size(), 0);

    auto sub = s.subspan(2, 3);
    ASSERT_EQ(sub.data(), a.data() + 2);
    ASSERT_EQ(sub.size(), 3);

    sub = s.subspan(100, 1);
    ASSERT_EQ(last.data(), nullptr);
    ASSERT_EQ(last.size(), 0);

    sub = s.subspan(0, 100);
    ASSERT_EQ(last.data(), nullptr);
    ASSERT_EQ(last.size(), 0);

    sub = s.subspan(6, 7);
    ASSERT_EQ(last.data(), nullptr);
    ASSERT_EQ(last.size(), 0);
}

TEST(span_test, SpanObservers)
{
    std::array<int32_t, 10> a;
    std::iota(a.begin(), a.end(), 0);

    aten::span<decltype(a)::value_type> s(a);
    ASSERT_EQ(s.data(), a.data());
    ASSERT_EQ(s.size(), a.size());

    ASSERT_EQ(s.size_bytes(), a.size() * sizeof(decltype(a)::value_type));

    aten::span<int32_t> s_empty;
    ASSERT_TRUE(s_empty.empty());
}

TEST(span_test, SpanAccess)
{
    std::array<int32_t, 10> a;
    std::iota(a.begin(), a.end(), 0);

    aten::span<decltype(a)::value_type> s(a);
    ASSERT_EQ(s.data(), a.data());
    ASSERT_EQ(s.size(), a.size());

    ASSERT_EQ(s[4], a[4]);
    ASSERT_EQ(s.front(), a.front());
    ASSERT_EQ(s.back(), a.back());
}

TEST(const_span_test, ConstSpanConstructor)
{
    aten::const_span<int32_t> s0;
    ASSERT_EQ(s0.data(), nullptr);
    ASSERT_EQ(s0.size(), 0);

    const int32_t x = 0;
    aten::const_span<decltype(x)> s1(&x, 1);
    ASSERT_EQ(s1.data(), &x);
    ASSERT_EQ(s1.size(), 1);

    const int32_t a[4] = { 0, 1, 2, 3 };
    using c_style_array_type = std::remove_reference_t<decltype(std::declval<decltype(a)>()[0])>;
    aten::const_span<c_style_array_type> s2(a);
    ASSERT_EQ(s2.data(), a);
    ASSERT_EQ(s2.size(), std::size(a));

    std::array<int32_t, 4> aa{ 0, 1, 2, 3 };
    aten::const_span<decltype(aa)::value_type> s3(aa);
    ASSERT_EQ(s3.data(), aa.data());
    ASSERT_EQ(s3.size(), aa.size());

    std::vector<int32_t> v{ 0, 1, 2, 3 };
    aten::const_span<decltype(aa)::value_type> s4(v);
    ASSERT_EQ(s4.data(), v.data());
    ASSERT_EQ(s4.size(), v.size());

    auto s5(s4);
    ASSERT_EQ(s5.data(), v.data());
    ASSERT_EQ(s5.size(), v.size());

    auto s6 = s5;
    ASSERT_EQ(s6.data(), v.data());
    ASSERT_EQ(s6.size(), v.size());
}

TEST(const_span_test, ConstSpanSubspane)
{
    std::array<int32_t, 10> a;
    std::iota(a.begin(), a.end(), 0);

    aten::const_span<decltype(a)::value_type> s(a);
    ASSERT_EQ(s.data(), a.data());
    ASSERT_EQ(s.size(), a.size());

    auto first = s.first(5);
    ASSERT_EQ(first.data(), a.data());
    ASSERT_EQ(first.size(), 5);

    first = s.first(100);
    ASSERT_EQ(first.data(), nullptr);
    ASSERT_EQ(first.size(), 0);

    auto last = s.last(4);
    ASSERT_EQ(last.data(), a.data() + (a.size() - 4));
    ASSERT_EQ(last.size(), 4);

    last = s.last(100);
    ASSERT_EQ(last.data(), nullptr);
    ASSERT_EQ(last.size(), 0);

    auto sub = s.subspan(2, 3);
    ASSERT_EQ(sub.data(), a.data() + 2);
    ASSERT_EQ(sub.size(), 3);

    sub = s.subspan(100, 1);
    ASSERT_EQ(last.data(), nullptr);
    ASSERT_EQ(last.size(), 0);

    sub = s.subspan(0, 100);
    ASSERT_EQ(last.data(), nullptr);
    ASSERT_EQ(last.size(), 0);

    sub = s.subspan(6, 7);
    ASSERT_EQ(last.data(), nullptr);
    ASSERT_EQ(last.size(), 0);
}

TEST(const_span_test, ConstSpanObservers)
{
    std::array<int32_t, 10> a;
    std::iota(a.begin(), a.end(), 0);

    aten::const_span<decltype(a)::value_type> s(a);
    ASSERT_EQ(s.data(), a.data());
    ASSERT_EQ(s.size(), a.size());

    ASSERT_EQ(s.size_bytes(), a.size() * sizeof(decltype(a)::value_type));

    aten::const_span<int32_t> s_empty;
    ASSERT_TRUE(s_empty.empty());
}

TEST(const_span_test, ConstSpanAccess)
{
    std::array<int32_t, 10> a;
    std::iota(a.begin(), a.end(), 0);

    aten::const_span<decltype(a)::value_type> s(a);
    ASSERT_EQ(s.data(), a.data());
    ASSERT_EQ(s.size(), a.size());

    ASSERT_EQ(s[4], a[4]);
    ASSERT_EQ(s.front(), a.front());
    ASSERT_EQ(s.back(), a.back());
}
