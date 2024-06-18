#include <gtest/gtest.h>

#include <array>
#include <numeric>
#include <type_traits>
#include <vector>

#include "aten.h"

TEST(stack_test, StackEmpty)
{
    aten::stack<int32_t, 5> stack;
    ASSERT_TRUE(stack.empty());

    stack.push(1);
    ASSERT_FALSE(stack.empty());

    stack.pop();
    ASSERT_TRUE(stack.empty());
}

TEST(stack_test, StackPushPop)
{
    aten::stack<int32_t, 5> stack;

    stack.push(1);
    ASSERT_EQ(stack.size(), 1);

    stack.push(2);
    ASSERT_EQ(stack.size(), 2);

    auto n = stack.pop();
    ASSERT_EQ(n, 2);
    ASSERT_EQ(stack.size(), 1);

    n = stack.pop();
    ASSERT_EQ(n, 1);
    ASSERT_EQ(stack.size(), 0);
}

TEST(stack_test, StackPushWithMove)
{
    struct Test {
        int32_t n{ 0 };
        bool construtor_by_moving{ false };

        Test() = default;

        Test& operator=(Test&& rhs)
        {
            n = rhs.n;
            construtor_by_moving = true;
            return *this;
        }
    };

    aten::stack<Test, 5> stack;

    Test data;
    data.n = 10;

    stack.push(std::move(data));
    ASSERT_EQ(stack.size(), 1);

    auto& res = stack.pop();
    ASSERT_EQ(stack.size(), 0);
    ASSERT_EQ(res.n, 10);
    ASSERT_TRUE(res.construtor_by_moving);

    // Check if the original data is not changed.
    ASSERT_FALSE(data.construtor_by_moving);
}

TEST(stack_test, StackTopBack)
{
    aten::stack<int32_t, 5> stack;

    stack.push(1);
    ASSERT_EQ(stack.size(), 1);

    auto n = stack.top();
    ASSERT_EQ(stack.size(), 1);
    ASSERT_EQ(n, 1);

    stack.push(2);
    ASSERT_EQ(stack.size(), 2);

    // front just returns the top value not pop.
    // So, stack size is the same.
    n = stack.top();
    ASSERT_EQ(stack.size(), 2);
    ASSERT_EQ(n, 1);

    // back just returns the bottom value not pop.
    // So, stack size is the same.
    n = stack.back();
    ASSERT_EQ(stack.size(), 2);
    ASSERT_EQ(n, 2);
}

TEST(stack_test, StackSafePop)
{
    aten::stack<int32_t, 5> stack;

    stack.push(1);
    ASSERT_EQ(stack.size(), 1);

    auto n = stack.pop();
    ASSERT_EQ(n, 1);
    ASSERT_EQ(stack.size(), 0);

    auto x = stack.safe_pop();
    ASSERT_FALSE(x.has_value());

    stack.push(2);
    ASSERT_EQ(stack.size(), 1);
    x = stack.safe_pop();
    ASSERT_TRUE(x.has_value());
    ASSERT_EQ(x.value(), 2);
}
