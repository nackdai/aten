#pragma once

#include "defs.h"

namespace AT_NAME
{
    AT_DEVICE_API inline real computeBalanceHeuristic(real f, real g)
    {
        return f / (f + g);
    }
}
