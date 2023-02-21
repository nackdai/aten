#pragma once

#include "visualizer/visualizer.h"
#include "visualizer/blitter.h"

namespace aten {
    class BilateralFilter: public visualizer::PreProc{
    public:
        BilateralFilter() {}
        BilateralFilter(real sigmaS, real sigmaR)
        {
            setParam(sigmaS, sigmaR);
        }

        virtual ~BilateralFilter() {}

    public:
        virtual void operator()(
            const vec4* src,
            uint32_t width, uint32_t height,
            vec4* dst) override final;

        void setParam(real sigmaS, real sigmaR)
        {
            m_sigmaS = sigmaS;
            m_sigmaR = sigmaR;
        }

        virtual void setParam(Values& values) override final
        {
            m_sigmaS = values.get("sigmas", m_sigmaS);
            m_sigmaR = values.get("sigmar", m_sigmaR);
        }

    private:
        real m_sigmaS{ real(0.2) };
        real m_sigmaR{ real(0.2) };
    };

    class BilateralFilterShader : public Blitter {
    public:
        BilateralFilterShader() {}
        BilateralFilterShader(real sigmaS, real sigmaR)
        {
            setParam(sigmaS, sigmaR);
        }

        virtual ~BilateralFilterShader() {}

    public:
        virtual void prepareRender(
            const void* pixels,
            bool revert) override final;

        void setParam(real sigmaS, real sigmaR)
        {
            m_sigmaS = sigmaS;
            m_sigmaR = sigmaR;
        }

        virtual void setParam(Values& values) override final
        {
            m_sigmaS = values.get("sigmas", m_sigmaS);
            m_sigmaR = values.get("sigmar", m_sigmaR);
        }

    private:
        real m_sigmaS{ real(0.2) };
        real m_sigmaR{ real(0.2) };

        static const uint32_t buffersize = 10;
        float distW[buffersize + 1][buffersize + 1];

        int32_t m_radius{ 0 };
    };
}
