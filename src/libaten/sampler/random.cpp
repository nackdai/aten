#include "sampler/random.h"
#include "sampler/halton.h"

namespace aten {
	void random::init()
	{
		// TODO
		::srand(0);

		Halton::makePrimeNumbers();
	}
}
