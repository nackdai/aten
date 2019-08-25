#pragma once

#include <cstdint>
#include <climits>

//#define TYPE_DOUBLE

#ifdef TYPE_DOUBLE
    using real = double;
    #define AT_IS_TYPE_DOUBLE    (true)
#else
    using real = float;
    #define AT_IS_TYPE_DOUBLE    (false)
#endif

