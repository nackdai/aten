#include "cuda/cudaTextureResource.h"

namespace idaten
{
	// NOTE
	// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/

	void CudaTextureResource::init(
		const aten::vec4* p,
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

	/////////////////////////////////////////////////////

	void CudaTexture::init(
		const aten::vec4* p,
		int width, int height)
	{
		// NOTE
		// http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?%A5%EA%A5%CB%A5%A2%A5%E1%A5%E2%A5%EA%A4%C8CUDA%C7%DB%CE%F3
		// http://www.orangeowlsolutions.com/archives/613

		// NOTE
		// 2Dテクスチャの場合は、pitchのアラインメントを考慮しないといけない.
		// cudaMallocPitch はアラインメントを考慮した処理になっている.

		size_t dstPitch = 0;
		size_t srcPitch = sizeof(float4) * width;

		checkCudaErrors(cudaMallocPitch(&m_buffer, &dstPitch, srcPitch, height));
		checkCudaErrors(cudaMemcpy2D(m_buffer, dstPitch, p, srcPitch, srcPitch, height, cudaMemcpyHostToDevice));

		// Make Resource description:
		memset(&m_resDesc, 0, sizeof(m_resDesc));
		m_resDesc.resType = cudaResourceTypePitch2D;
		m_resDesc.res.pitch2D.devPtr = m_buffer;
		m_resDesc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
		m_resDesc.res.pitch2D.desc.x = 32; // bits per channel
		m_resDesc.res.pitch2D.desc.y = 32; // bits per channel
		m_resDesc.res.pitch2D.desc.z = 32; // bits per channel
		m_resDesc.res.pitch2D.desc.w = 32; // bits per channel
		m_resDesc.res.pitch2D.width = width;
		m_resDesc.res.pitch2D.height = height;
		m_resDesc.res.pitch2D.pitchInBytes = dstPitch;
	}

	cudaTextureObject_t CudaTexture::bind()
	{
		if (m_buffer) {
			// TODO
			// Only for resource array.

			// Make texture description:
			cudaTextureDesc tex_desc = {};
			tex_desc.readMode = cudaReadModeElementType;
			tex_desc.filterMode = cudaFilterModeLinear;
			tex_desc.addressMode[0] = cudaAddressModeWrap;
			tex_desc.addressMode[1] = cudaAddressModeWrap;
			tex_desc.normalizedCoords = 1;

			checkCudaErrors(cudaCreateTextureObject(&m_tex, &m_resDesc, &tex_desc, nullptr));
		}

		return m_tex;
	}
}