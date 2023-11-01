#include <gtest/gtest.h>

#include <array>
#include <numeric>
#include <type_traits>
#include <vector>

#include "aten.h"

namespace _test_detail {
    template<class T>
    using HasFuncOp = decltype(std::declval<T>().func());
}

TEST(misc_test, IsDetected)
{
    struct Foo {
        void func() {}
    };

    Foo foo;

    constexpr auto b0 = aten::is_detected<_test_detail::HasFuncOp, Foo>::value;
    ASSERT_TRUE(b0);

    constexpr auto b1 = aten::is_detected<_test_detail::HasFuncOp, decltype(foo)>::value;
    ASSERT_TRUE(b1);

    constexpr auto b2 = aten::is_detected<_test_detail::HasFuncOp, int>::value;
    ASSERT_FALSE(b2);
}
