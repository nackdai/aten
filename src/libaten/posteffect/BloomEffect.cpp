#include "visualizer/atengl.h"
#include "posteffect/BloomEffect.h"

namespace aten {
    bool BloomEffect::BloomEffectPass::init(
        int srcWidth, int srcHeight,
        int dstWidth, int dstHeight,
        PixelFormat inFmt, PixelFormat outFmt,
        const char* pathVS,
        const char* pathFS)
    {
        m_srcWidth = srcWidth;
        m_srcHeight = srcHeight;

        m_fmtIn = inFmt;
        m_fmtOut = outFmt;

        bool result = shader::init(dstWidth, dstHeight, pathVS, pathFS);
        return result;
    }

    void BloomEffect::BloomEffectPass::prepareRender(
        const void* pixels,
        bool revert)
    {
        shader::prepareRender(pixels, revert);

        CALL_GL_API(glUniform1i(getHandle("image"), 0));

        float srcTexel[2] = { 1.0f / m_srcWidth, 1.0f / m_srcHeight };

        float dstTexel[2] = { 1.0f / m_width, 1.0f / m_height };

        CALL_GL_API(glUniform2f(
            getHandle("srcTexel"),
            srcTexel[0], srcTexel[1]));

        CALL_GL_API(glUniform2f(
            getHandle("dstTexel"),
            dstTexel[0], dstTexel[1]));

        // TODO
        {
            auto hThreshold = getHandle("threshold");
            if (hThreshold >= 0) {
                auto threshold = m_body->m_threshold;
                CALL_GL_API(glUniform1f(hThreshold, threshold));
            }

            auto hAdaptedLum = getHandle("adaptedLum");
            if (hAdaptedLum >= 0) {
                auto adaptedLum = m_body->m_adaptedLum;
                CALL_GL_API(glUniform1f(hThreshold, adaptedLum));
            }
        }
    }

    void BloomEffect::BloomEffectFinalPass::prepareRender(
        const void* pixels,
        bool revert)
    {
        shader::prepareRender(pixels, revert);

        // Source tex handle.
        {
            GLuint srcTexHandle = m_body->getVisualizer()->getTexHandle();
            auto prevEffectPass = m_body->getPrevPass();
            if (prevEffectPass) {
                srcTexHandle = prevEffectPass->getFbo().getTexHandle();
            }

            // Bind texture.
            CALL_GL_API(glActiveTexture(GL_TEXTURE0));

            CALL_GL_API(glBindTexture(GL_TEXTURE_2D, srcTexHandle));

            // Specify filter after binding!!!!!
            CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP));
            CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP));
            CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
            CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        }

        // Bloomed tex handle.
        {
            auto prevPass = getPrevPass();
            auto texHandle = prevPass->getFbo().getTexHandle();

            // Bind texture.
            CALL_GL_API(glActiveTexture(GL_TEXTURE1));

            CALL_GL_API(glBindTexture(GL_TEXTURE_2D, texHandle));

            // Specify filter after binding!!!!!
            CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP));
            CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP));
            CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
            CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        }

        CALL_GL_API(glUniform1i(getHandle("image"), 0));
        CALL_GL_API(glUniform1i(getHandle("bloomtex"), 1));

        float texel[2] = { 1.0f / m_width, 1.0f / m_height };

        CALL_GL_API(glUniform2f(
            getHandle("texel"),
            texel[0], texel[1]));
    }

    bool BloomEffect::init(
        int width, int height,
        PixelFormat inFmt, PixelFormat outFmt,
        const char* pathVS,
        const char* pathFS_4x4,
        const char* pathFS_2x2,
        const char* pathFS_HBlur,
        const char* pathFS_VBlur,
        const char* pathFS_GaussBlur,
        const char* pathFS_Final)
    {
        m_fmtIn = inFmt;
        m_fmtOut = outFmt;

        auto windowWidth = width;
        auto windowHeight = height;

        auto scaledWidth = width / 4;
        auto scaledHeight = height / 4;

        auto scaledWidth_2 = scaledWidth / 2;
        auto scaledHeight_2 = scaledHeight / 2;

        AT_VRETURN(
            m_pass4x4.init(
                windowWidth, windowHeight,
                scaledWidth, scaledHeight,
                m_fmtIn, m_fmtOut,
                pathVS, pathFS_4x4), false);

        AT_VRETURN(
            m_passGaussBlur_4x4.init(
                scaledWidth, scaledHeight,
                scaledWidth, scaledHeight,
                m_fmtOut, m_fmtOut,
                pathVS, pathFS_GaussBlur), false);

        AT_VRETURN(
            m_pass2x2.init(
                scaledWidth, scaledHeight,
                scaledWidth_2, scaledHeight_2,
                m_fmtOut, m_fmtOut,
                pathVS, pathFS_2x2), false);

        AT_VRETURN(
            m_passGaussBlur_2x2.init(
                scaledWidth_2, scaledHeight_2,
                scaledWidth_2, scaledHeight_2,
                m_fmtOut, m_fmtOut,
                pathVS, pathFS_GaussBlur), false);

        AT_VRETURN(
            m_passVBlur.init(
                scaledWidth_2, scaledHeight_2,
                scaledWidth_2, scaledHeight_2,
                m_fmtOut, m_fmtOut,
                pathVS, pathFS_VBlur), false);

        AT_VRETURN(
            m_passHBlur.init(
                scaledWidth_2, scaledHeight_2,
                scaledWidth_2, scaledHeight_2,
                m_fmtOut, m_fmtOut,
                pathVS, pathFS_HBlur), false);

        // For final pass.
        AT_VRETURN(
            m_passFinal.init(
                windowWidth, windowHeight,
                windowWidth, windowHeight,
                m_fmtOut, m_fmtOut,
                pathVS, pathFS_Final), false);

        addPass(&m_pass4x4);
        addPass(&m_passGaussBlur_4x4);
        addPass(&m_pass2x2);
        addPass(&m_passVBlur);
        addPass(&m_passHBlur);
        addPass(&m_passFinal);

        return true;
    }
}
