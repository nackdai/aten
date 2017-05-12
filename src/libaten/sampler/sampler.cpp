#include "sampler/samplerinterface.h"
#include "sampler/halton.h"

namespace aten {
	void initSampler()
	{
		// TODO
		::srand(0);

		Halton::makePrimeNumbers();
	}
}
