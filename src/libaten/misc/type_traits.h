#pragma once

#include <type_traits>

// NOTE:
// https://qiita.com/tyanmahou/items/c2a7c389e666d1b1bdea
// Alternative implementation for std::experimental::is_detected.

namespace aten {
    namespace _detail
    {
        // NOTE:
        // https://onihusube.hatenablog.com/entry/2019/01/29/004929
        // https://stackoverflow.com/questions/27687389/how-do-we-use-void-t-for-sfinae
        // https://stackoverflow.com/questions/59276542/sfinae-understanding-void-t-and-detect-if

        // This is the primary class template.
        // If no specialization matches, the definition of the primary template is used as a fall-back.
        template<class AlwaysVoid, template<class...>class Op, class ...Args>
        struct detector
            : std::false_type {};

        // This the partial specialized class template.
        // Even though the primary template is specified, the specialization is deduced firstly.
        // If std::void_t<Op<Args...>> is deduced and it's the well-formed, this specialization is deduced.
        // It means the deduced template arguments are substituted and this specialization is used.
        // But, std::void_t<Op<Args...>> is the ill-formed, the specialization isn't used and fall-back to the primary template happens.
        template<template<class...>class Op, class ...Args>
        struct detector<std::void_t<Op<Args...>>, Op, Args...>
            : std::true_type {};
    }

    // How to use:
    // template<class T>
    // using HasFuncOp = decltype(std::declval<T>().func());
    //
    // template<class T>
    // using HasFunc = is_detected<HasFuncOp, T>;
    //
    // HasFuncOp in HasFunc is expanded as HasFuncOp<T> in is_detected.
    // And then, it is treated as std::void_t<Op<Args...>>. In this case, Op is HasFuncOp and Args... is T.
    // So, std::void_t<Op<Args...>> is std::void_t<HasFuncOp<T>>.
    // If it's available, it's fallen for std::true_type side. Otherwise, std::false_type side.

    // NOTE:
    // detector<void, Op, Args...> seems to specify the primary template.
    // But, even though the primary template is specified, the specialization is deduced firstly.
    template<template<class...>class Op, class ...Args>
    using is_detected = _detail::detector<void, Op, Args...>;

    template <class T>
    struct is_shared_ptr : std::false_type {};

    // If is_shared_ptr is used with std::shared_ptr, like is_shared_ptr<std::shared_ptr<blah>>.
    // Fall back to this template specialization.
    template <class T>
    struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

    template <class T>
    inline constexpr bool is_shared_ptr_v = is_shared_ptr<T>::value;
}
