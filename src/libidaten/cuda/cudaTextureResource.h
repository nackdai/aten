#pragma once

#include "aten4idaten.h"
#include "cuda/cudautil.h"

#include <cuda_runtime.h>

namespace idaten
{
	class CudaTextureResource {
	public:
		CudaTextureResource() {}
		~CudaTextureResource() {}

	public:
		void init(
			aten::vec4* p, 
			uint32_t memberNumInItem, 
			uint32_t numOfContaints);

		cudaTextureObject_t bind();
		void unbind();

	protected:
		void* m_buffer{ nullptr };
		cudaResourceDesc m_resDesc;
		cudaTextureObject_t m_tex{ 0 };
	};
}