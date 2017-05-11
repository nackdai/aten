#include "sampler/sampler.h"
#include "sampler/halton.h"

namespace aten {
	void sampler::init()
	{
		// TODO
		::srand(0);

		Halton::makePrimeNumbers();
	}
}
