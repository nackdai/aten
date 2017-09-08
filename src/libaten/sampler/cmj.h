#pragma once

#include "types.h"
#include "sampler/samplerinterface.h"

namespace aten {
	// Correllated multi jittered.
	class CMJ : public sampler {
	public:
		AT_DEVICE_API CMJ() {}
		AT_VIRTUAL(AT_DEVICE_API ~CMJ() {});

	public:
		AT_VIRTUAL_OVERRIDE_FINAL(AT_DEVICE_API void init(uint32_t seed, const unsigned int* data = nullptr))
		{
			// TODO
		}

		AT_DEVICE_API void init(
			uint32_t index,
			uint32_t dimension,
			uint32_t scramble,
			const unsigned int* data = nullptr)
		{
			// TODO
		}

		AT_VIRTUAL_OVERRIDE_FINAL(AT_DEVICE_API real nextSample())
		{
			// TODO
			return 0.0f;
		}

	private:
		uint32_t permute(uint32_t i, uint32_t l, uint32_t p)
		{
			uint32_t w = l - 1;
			w |= w >> 1;
			w |= w >> 2;
			w |= w >> 4;
			w |= w >> 8;
			w |= w >> 16;

			do
			{
				i ^= p;
				i *= 0xe170893d;
				i ^= p >> 16;
				i ^= (i & w) >> 4;
				i ^= p >> 8;
				i *= 0x0929eb3f;
				i ^= p >> 23;
				i ^= (i & w) >> 1;
				i *= 1 | p >> 27;
				i *= 0x6935fa69;
				i ^= (i & w) >> 11;
				i *= 0x74dcb303;
				i ^= (i & w) >> 2;
				i *= 0x9e501cc3;
				i ^= (i & w) >> 2;
				i *= 0xc860a3df;
				i &= w;
				i ^= i >> 5;
			} while (i >= l);

			return (i + p) % l;
		}

		float randfloat(uint32_t i, uint32_t p)
		{
			i ^= p;
			i ^= i >> 17;
			i ^= i >> 10;
			i *= 0xb36534e5;
			i ^= i >> 12;
			i ^= i >> 21;
			i *= 0x93fc4795;
			i ^= 0xdf6e307f;
			i ^= i >> 17;
			i *= 1 | p >> 18;

			return i * (1.0f / 4294967808.0f);
		}

	private:
		uint32_t m_idx;
		uint32_t m_dimension;
		uint32_t m_scramble;
	};
}
