#pragma once

#ifdef __AT_CUDA__

#undef AT_VIRTUAL
#undef AT_VIRTUAL_OVERRIDE
#undef AT_VIRTUAL_OVERRIDE_FINAL
#undef AT_PURE_VIRTUAL
#undef AT_INHERIT

#define AT_VIRTUAL(f)					f
#define AT_VIRTUAL_OVERRIDE(f)			f
#define AT_VIRTUAL_OVERRIDE_FINAL(f)	f
#define AT_PURE_VIRTUAL(f)				f
#define AT_INHERIT(c)

#include "sampler/wanghash.h"
#include "sampler/sobolproxy.h"

#define IDATEN_USE_SOBOL

namespace aten {
#ifdef IDATEN_USE_SOBOL
	using sampler = Sobol;
#else
	using sampler = WangHash;
#endif
}

#include "aten_virtual.h"
#else
#include "samplerinterface.h"
#endif