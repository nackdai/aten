#include <string.h>
#include "cuda/cudaGLresource.h"

namespace idaten
{
    static const uint32_t flags[] = {
        cudaGraphicsRegisterFlagsNone,
        cudaGraphicsRegisterFlagsReadOnly,
        cudaGraphicsRegisterFlagsWriteDiscard,
    };

    void CudaGLSurface::init(GLuint gltex, CudaGLRscRegisterType type)
    {
        AT_ASSERT(m_gltex == 0);
        AT_ASSERT(!m_rsc);

        if (!m_rsc) {
            checkCudaErrors(cudaGraphicsGLRegisterImage(
                &m_rsc,
                gltex,
                GL_TEXTURE_2D,
                flags[type]));

            m_gltex = gltex;
        }
    }

    cudaSurfaceObject_t CudaGLSurface::bind()
    {
        if (m_surf > 0) {
            return m_surf;
        }

        AT_ASSERT(m_rsc);

        // NOTE
        // cudaGraphicsSubResourceGetMappedArray has to be called after resource is mapped.
        AT_ASSERT(m_isMapped);

        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&m_cudaArray, m_rsc, 0, 0));

        m_surf = 0;

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = m_cudaArray;

        checkCudaErrors(cudaCreateSurfaceObject(&m_surf, &resDesc));

        return m_surf;
    }

    void CudaGLSurface::unbind()
    {
        AT_ASSERT(m_isMapped);

        if (m_surf > 0) {
            checkCudaErrors(cudaDestroySurfaceObject(m_surf));
            m_surf = 0;
        }
    }

    //////////////////////////////////////////////////////////////////////////

    void CudaGLBuffer::init(GLuint glbuffer, CudaGLRscRegisterType type)
    {
        AT_ASSERT(m_glbuffer == 0);
        AT_ASSERT(!m_rsc);

        if (!m_rsc) {
            checkCudaErrors(cudaGraphicsGLRegisterBuffer(
                &m_rsc,
                glbuffer,
                flags[type]));

            m_glbuffer = glbuffer;
        }
    }

    void CudaGLBuffer::bind(void** p, size_t& bytes)
    {
        AT_ASSERT(m_rsc);
        AT_ASSERT(p);

        checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
            (void**)p,
            &bytes,
            m_rsc));
    }

    void CudaGLBuffer::unbind()
    {
        // Nothing is done...
    }
}
