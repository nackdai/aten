#include "accelerator/accelerator.h"
#include "accelerator/bvh.h"

namespace aten {
	std::tuple<accelerator*, int> accelerator::createAccelerator(bool needRegister/*= false*/)
	{
		// TODO
		auto ret = new bvh();

		int id = -1;
		if (needRegister) {
			id = bvh::registerToList(ret);
		}

		return std::make_tuple(ret, id);
	}
}
