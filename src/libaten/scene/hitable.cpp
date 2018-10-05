#include "scene/hitable.h"
#include "accelerator/accelerator.h"

namespace aten
{
    accelerator* hitable::getInternalAccelerator()
    {
        return nullptr;
    }
}
