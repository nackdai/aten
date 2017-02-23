#pragma once

#include <vector>
#include "types.h"
#include "sampler/random.h"

namespace aten {
	class Halton : public random {
	private:
		static std::vector<uint32_t> PrimeNumbers;

	public:
		static const uint32_t MaxPrimeNumbers = 10000000;

		// ëfêîê∂ê¨.
		static void makePrimeNumbers(uint32_t maxNumber = MaxPrimeNumbers);

	public:
		Halton(uint32_t idx) {
			m_idx = (idx == 0 ? 1 : idx);
		}
		virtual ~Halton() {}

		// [0, 1]
		virtual real next01() override final;

	private:
		uint32_t m_idx{ 1 };
		uint32_t m_dimension{ 0 };
	};
}
