#include "accelerator/accelerator.h"
#include "accelerator/bvh.h"
#include "accelerator/sbvh.h"

namespace aten {
	accelerator::AccelType accelerator::s_internalType = accelerator::AccelType::Bvh;
	std::function<accelerator*()> accelerator::s_userDefsInternalAccelCreator = nullptr;

	void accelerator::setInternalAccelType(AccelType type)
	{
		s_internalType = type;
	}

	accelerator::AccelType accelerator::getInternalAccelType()
	{
		return s_internalType;
	}

	void accelerator::setUserDefsInternalAccelCreator(std::function<accelerator*()> creator)
	{
		if (creator) {
			s_userDefsInternalAccelCreator = creator;
			s_internalType = AccelType::UserDefs;
		}
	}

	accelerator* accelerator::createAccelerator()
	{
		accelerator* ret = nullptr;

		if (s_internalType == AccelType::UserDefs
			&& s_userDefsInternalAccelCreator)
		{
			ret = s_userDefsInternalAccelCreator();
		}
		else {
			switch (s_internalType) {
			case AccelType::Sbvh:
				ret = new sbvh();
				break;
			default:
				ret = new bvh();
				break;
			}
		}

		return ret;
	}
}
