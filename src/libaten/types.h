#pragma once

#include <stdint.h>
#include <climits>

//#define TYPE_DOUBLE

#ifdef TYPE_DOUBLE
	using real = double;
#else
	using real = float;
#endif

