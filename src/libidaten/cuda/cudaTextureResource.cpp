#include "cuda/cudaTextureResource.h"

namespace idaten
{
	// NOTE
	// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/

	void CudaTextureResource::init(
		aten::vec4* p,
		uint32_t memberNumInItem,
		uint32_t numOfContaints)
	{
		auto size = sizeof(float4) * memberNumInItem * numOfContaints;

		checkCudaErrors(cudaMalloc(&m_buffer, size));
		checkCudaErrors(cudaMemcpy(m_buffer, p, size, cudaMemcpyHostToDevice));

		// Make Resource description:
		memset(&m_resDesc, 0, sizeof(m_resDesc));
		m_resDesc.resType = cudaResourceTypeLinear;
		m_resDesc.res.linear.devPtr = m_buffer;
		m_resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
		m_resDesc.res.linear.desc.x = 32; // bits per channel
		m_resDesc.res.linear.desc.y = 32; // bits per channel
		m_resDesc.res.linear.desc.z = 32; // bits per channel
		m_resDesc.res.linear.desc.w = 32; // bits per channel
		m_resDesc.res.linear.sizeInBytes = memberNumInItem * numOfContaints * sizeof(float4);
	}

	cudaTextureObject_t CudaTextureResource::bind()
	{
		if (m_buffer) {
			// TODO
			// Only for resource array.

			// Make texture description:
			cudaTextureDesc tex_desc = {};
			tex_desc.readMode = cudaReadModeElementType;
			tex_desc.filterMode = cudaFilterModePoint;
			tex_desc.addressMode[0] = cudaAddressModeClamp;
			tex_desc.addressMode[1] = cudaAddressModeClamp;
			tex_desc.normalizedCoords = 0;

			checkCudaErrors(cudaCreateTextureObject(&m_tex, &m_resDesc, &tex_desc, nullptr));
		}

		return m_tex;
	}

	void CudaTextureResource::unbind()
	{
		if (m_tex > 0) {
			checkCudaErrors(cudaDestroyTextureObject(m_tex));
			m_tex = 0;
		}
	}
}