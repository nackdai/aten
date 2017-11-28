#include "accelerator/accelerator.h"
#include "accelerator/bvh.h"
#include "accelerator/sbvh.h"

namespace aten {
	accelerator::AccelType accelerator::s_internalType = accelerator::AccelType::Bvh;

	void accelerator::setInternalAccelType(AccelType type)
	{
		s_internalType = type;
	}

	accelerator::AccelType accelerator::getInternalAccelType()
	{
		return s_internalType;
	}

	accelerator* accelerator::createAccelerator()
	{
		accelerator* ret = nullptr;

		switch (s_internalType) {
		case AccelType::Sbvh:
			ret = new sbvh();
			break;
		default:
			ret = new bvh();
			break;
		}

		return ret;
	}
}
