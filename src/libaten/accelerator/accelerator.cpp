#include "accelerator/accelerator.h"
#include "accelerator/bvh.h"
#include "accelerator/sbvh.h"
#include "accelerator/threaded_bvh.h"

namespace aten {
    AccelType accelerator::s_internalType = AccelType::Bvh;
    std::function<std::unique_ptr<accelerator>()> accelerator::s_userDefsInternalAccelCreator = nullptr;

    void accelerator::setInternalAccelType(AccelType type)
    {
        s_internalType = type;
    }

    AccelType accelerator::getInternalAccelType()
    {
        return s_internalType;
    }

    void accelerator::setUserDefsInternalAccelCreator(
        std::function<std::unique_ptr<accelerator>()> creator)
    {
        if (creator) {
            s_userDefsInternalAccelCreator = creator;
        }
    }

    std::unique_ptr<accelerator> accelerator::createAccelerator(AccelType type/*= AccelType::Default*/)
    {
        std::unique_ptr<accelerator> ret;

        type = (type == AccelType::Default ? s_internalType : type);

        if (type == AccelType::UserDefs)
        {
            AT_ASSERT(s_userDefsInternalAccelCreator);
            ret = s_userDefsInternalAccelCreator();
        }
        else {
            switch (type) {
            case AccelType::Sbvh:
                ret = std::make_unique<sbvh>();
                break;
            case AccelType::ThreadedBvh:
                ret = std::make_unique<ThreadedBVH>();
                break;
            default:
                ret = std::make_unique<bvh>();
                AT_ASSERT(false);
                break;
            }
        }

        return std::move(ret);
    }
}
