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
        // Even though the primary template is specified, the specialization is deducted firstly.
        // If std::void_t<Op<Args...>> is deducted and it's the well-formed, this specialization is deducted.
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
    // But, even though the primary template is specified, the specialization is deducted firstly.
    template<template<class...>class Op, class ...Args>
    using is_detected = typename _detail::detector<void, Op, Args...>;
}
