#pragma once

#include <vector>
#include "defs.h"
#include "visualizer/visualizer.h"

namespace aten {
    class MultiPassPostProc : public visualizer::PostProc {
    protected:
        MultiPassPostProc() {}
        virtual ~MultiPassPostProc() {}

    protected:
        virtual void PrepareRender(
            const void* pixels,
            bool revert) override;

        bool addPass(visualizer::PostProc* pass);

    private:
        std::vector<visualizer::PostProc*> m_passes;
    };
}
