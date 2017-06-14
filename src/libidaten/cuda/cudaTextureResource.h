#pragma once

#include "aten4idaten.h"
#include "cuda/cudautil.h"

#include <cuda_runtime.h>

namespace idaten
{
	class CudaTextureResource {
	public:
		CudaTextureResource() {}
		virtual ~CudaTextureResource() {}

	public:
		void init(
			const aten::vec4* p,
			uint32_t memberNumInItem, 
			uint32_t numOfContaints);

		virtual cudaTextureObject_t bind();
		void unbind();

	protected:
		void* m_buffer{ nullptr };
		cudaResourceDesc m_resDesc;
		cudaTextureObject_t m_tex{ 0 };
	};

	struct TextureResource {
		const aten::vec4* ptr;
		int width;
		int height;

		TextureResource(const aten::vec4* p, int w, int h)
			: ptr(p), width(w), height(h)
		{}
	};

	class CudaTexture : public CudaTextureResource {
	public:
		CudaTexture() {}
		~CudaTexture() {}

	public:
		void init(
			const aten::vec4* p,
			int width, int height);

		virtual cudaTextureObject_t bind() override final;
	};
}