#pragma once

#ifdef __AT_DEBUG__
#define AT_CUDA_INLINE
#else
#define AT_CUDA_INLINE	inline
#endif