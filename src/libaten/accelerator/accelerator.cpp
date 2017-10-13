#include "accelerator/accelerator.h"
#include "accelerator/bvh.h"

namespace aten {
	accelerator* accelerator::createAccelerator()
	{
		// TODO
		auto ret = new bvh();

		return ret;
	}
}
