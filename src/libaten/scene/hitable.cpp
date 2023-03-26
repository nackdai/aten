#include "accelerator/accelerator.h"
#include "scene/hitable.h"

namespace aten
{
    accelerator* hitable::getInternalAccelerator()
    {
        return nullptr;
    }
}
