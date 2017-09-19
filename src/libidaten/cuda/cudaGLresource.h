#pragma once

#include "defs.h"
#include "cuda/cudautil.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace idaten
{
	enum CudaGLRscRegisterType {
		ReadWrite,
		ReadOnly,
		WriteOnly,
	};

	class CudaGLResource {
	protected:
		CudaGLResource() {}
		virtual ~CudaGLResource()
		{
			if (m_rsc) {
				//checkCudaErrors(cudaGraphicsUnregisterResource(m_rsc));
			}
		}

	public:
		virtual void map()
		{
			AT_ASSERT(m_rsc);
			AT_ASSERT(!m_isMapped);
			if (!m_isMapped) {
				checkCudaErrors(cudaGraphicsMapResources(1, &m_rsc, 0));
				m_isMapped = true;
			}
		}

		virtual void unmap()
		{
			AT_ASSERT(m_rsc);
			AT_ASSERT(m_isMapped);
			if (m_isMapped) {
				checkCudaErrors(cudaGraphicsUnmapResources(1, &m_rsc, 0));
				m_isMapped = false;
			}
		}

	protected:
		cudaGraphicsResource_t m_rsc{ nullptr };
		bool m_isMapped{ false };
	};

	class CudaGLSurface : public CudaGLResource {
	public:
		CudaGLSurface() {}
		CudaGLSurface(GLuint gltex, CudaGLRscRegisterType type)
		{
			init(gltex, type);
		}
		~CudaGLSurface() {}

	public:
		void init(GLuint gltex, CudaGLRscRegisterType type);

		bool isValid() const
		{
			return (m_gltex > 0);
		}

		cudaSurfaceObject_t bind();
		void unbind();

		virtual void unmap() override final
		{
			unbind();
			CudaGLResource::unmap();
		}

	private:
		GLuint m_gltex{ 0 };
		cudaArray* m_cudaArray;
		cudaSurfaceObject_t m_surf{ 0 };
	};

	class CudaGLResourceMap {
	public:
		CudaGLResourceMap(CudaGLResource* rsc)
		{
			m_rsc = rsc;
			m_rsc->map();
		}
		~CudaGLResourceMap()
		{
			if (m_rsc) {
				m_rsc->unmap();
			}
		}

	private:
		CudaGLResource* m_rsc;
	};
}