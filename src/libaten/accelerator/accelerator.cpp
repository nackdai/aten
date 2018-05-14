#include "accelerator/accelerator.h"
#include "accelerator/bvh.h"
#include "accelerator/sbvh.h"
#include "accelerator/threaded_bvh.h"

namespace aten {
	AccelType accelerator::s_internalType = AccelType::Bvh;
	std::function<accelerator*()> accelerator::s_userDefsInternalAccelCreator = nullptr;

	void accelerator::setInternalAccelType(AccelType type)
	{
		s_internalType = type;
	}

	AccelType accelerator::getInternalAccelType()
	{
		return s_internalType;
	}

	void accelerator::setUserDefsInternalAccelCreator(std::function<accelerator*()> creator)
	{
		if (creator) {
			s_userDefsInternalAccelCreator = creator;
		}
	}

	accelerator* accelerator::createAccelerator(AccelType type/*= AccelType::Default*/)
	{
		accelerator* ret = nullptr;

		type = (type == AccelType::Default ? s_internalType : type);

		if (type == AccelType::UserDefs)
		{
			AT_ASSERT(s_userDefsInternalAccelCreator);
			ret = s_userDefsInternalAccelCreator();
		}
		else {
			switch (type) {
			case AccelType::Sbvh:
				ret = new sbvh();
				break;
			case AccelType::ThreadedBvh:
				ret = new ThreadedBVH();
				break;
			default:
				ret = new bvh();
				AT_ASSERT(false);
				break;
			}
		}

		return ret;
	}
}
