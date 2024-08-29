#include <gtest/gtest.h>

#include <array>
#include <numeric>
#include <type_traits>
#include <vector>

#include "aten.h"

namespace _test_detail {
    template<class T>
    using HasFuncOp = decltype(std::declval<T>().func());

    template<class T, class N>
    using HasFuncOpWithArgs = decltype(std::declval<T>().func_arg(std::declval<N>()));
}

TEST(misc_test, IsDetected)
{
    struct Foo {
        void func() {}

        void func_arg(int32_t n) {}
    };

    Foo foo;

    constexpr auto b0 = aten::is_detected<_test_detail::HasFuncOp, Foo>::value;
    ASSERT_TRUE(b0);

    constexpr auto b1 = aten::is_detected<_test_detail::HasFuncOp, decltype(foo)>::value;
    ASSERT_TRUE(b1);

    constexpr auto b2 = aten::is_detected<_test_detail::HasFuncOp, int>::value;
    ASSERT_FALSE(b2);

    constexpr auto b3 = aten::is_detected<_test_detail::HasFuncOpWithArgs, Foo, int32_t>::value;
    ASSERT_TRUE(b3);
}

TEST(misc_test, IsSharedPtr)
{
    constexpr auto b0 = aten::is_shared_ptr_v<std::shared_ptr<int>>;
    ASSERT_TRUE(b0);

    constexpr auto b1 = aten::is_shared_ptr_v<int>;
    ASSERT_FALSE(b1);
}
