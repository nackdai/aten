#include "cuda/cudaTextureResource.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

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
		if (m_tex == 0) {
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
			//checkCudaErrors(cudaDestroyTextureObject(m_tex));
			//m_tex = 0;
		}
	}

	void CudaTextureResource::update(
		const aten::vec4* p,
		uint32_t memberNumInItem,
		uint32_t numOfContaints)
	{
		auto size = sizeof(float4) * memberNumInItem * numOfContaints;

		checkCudaErrors(cudaMemcpy(m_buffer, p, size, cudaMemcpyHostToDevice));
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
		if (m_tex == 0) {
			if (m_isMipmap) {
				cudaTextureDesc tex_desc = {};

				tex_desc.normalizedCoords = 1;
				tex_desc.filterMode = cudaFilterModeLinear;
				tex_desc.mipmapFilterMode = cudaFilterModeLinear;

#if 0
				tex_desc.addressMode[0] = cudaAddressModeClamp;
				tex_desc.addressMode[1] = cudaAddressModeClamp;
				tex_desc.addressMode[2] = cudaAddressModeClamp;
#else
				tex_desc.addressMode[0] = cudaAddressModeWrap;
				tex_desc.addressMode[1] = cudaAddressModeWrap;
				tex_desc.addressMode[2] = cudaAddressModeWrap;
#endif

				tex_desc.maxMipmapLevelClamp = float(m_mipmapLevel - 1);

				tex_desc.readMode = cudaReadModeElementType;

				checkCudaErrors(cudaCreateTextureObject(&m_tex, &m_resDesc, &tex_desc, nullptr));
			}
			else {
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
		}

		return m_tex;
	}

	////////////////////////////////////////////////////////////////////////////////////

	// NOTE
	// http://www.cse.uaa.alaska.edu/~ssiewert/a490dmis_code/CUDA/cuda_work/samples/2_Graphics/bindlessTexture/bindlessTexture_kernel.cu

	__global__ void genMipmap(
		cudaSurfaceObject_t mipOutput, 
		cudaTextureObject_t mipInput, 
		int imageW, int imageH)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		float px = 1.0 / float(imageW);
		float py = 1.0 / float(imageH);

		if ((x < imageW) && (y < imageH))
		{
			// take the average of 4 samples

			// we are using the normalized access to make sure non-power-of-two textures
			// behave well when downsized.
			float4 color =
				(tex2D<float4>(mipInput, (x + 0) * px, (y + 0) * py)) +
				(tex2D<float4>(mipInput, (x + 1) * px, (y + 0) * py)) +
				(tex2D<float4>(mipInput, (x + 1) * px, (y + 1) * py)) +
				(tex2D<float4>(mipInput, (x + 0) * px, (y + 1) * py));


			color /= 4.0f;

			surf2Dwrite(color, mipOutput, x * sizeof(float4), y);
		}
	}

	static void onGenMipmaps(
		cudaMipmappedArray_t mipmapArray,
		int width, int height,
		int maxLevel)
	{
		int level = 0;

		//while (width != 1 || height != 1)
		while (level + 1 < maxLevel)
		{
			width /= 2;
			height /= 2;

			width = std::max(1, width);
			height = std::max(1, height);

			// Copy from.
			cudaArray_t levelFrom;
			checkCudaErrors(cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level));

			// Copy to.
			cudaArray_t levelTo;
			checkCudaErrors(cudaGetMipmappedArrayLevel(&levelTo, mipmapArray, level + 1));

			cudaExtent levelToSize;
			checkCudaErrors(cudaArrayGetInfo(nullptr, &levelToSize, nullptr, levelTo));
			AT_ASSERT(levelToSize.width == width);
			AT_ASSERT(levelToSize.height == height);
			AT_ASSERT(levelToSize.depth == 0);

			// generate texture object for reading
			cudaTextureObject_t texInput;
			{
				cudaResourceDesc texRes;
				{
					memset(&texRes, 0, sizeof(cudaResourceDesc));

					texRes.resType = cudaResourceTypeArray;
					texRes.res.array.array = levelFrom;
				}

				cudaTextureDesc texDesc;
				{
					memset(&texDesc, 0, sizeof(cudaTextureDesc));

					texDesc.normalizedCoords = 1;
					texDesc.filterMode = cudaFilterModeLinear;
					texDesc.addressMode[0] = cudaAddressModeClamp;
					texDesc.addressMode[1] = cudaAddressModeClamp;
					texDesc.addressMode[2] = cudaAddressModeClamp;
					texDesc.readMode = cudaReadModeElementType;
				}

				checkCudaErrors(cudaCreateTextureObject(&texInput, &texRes, &texDesc, nullptr));
			}

			// generate surface object for writing
			cudaSurfaceObject_t surfOutput;
			{
				cudaResourceDesc surfRes;
				{
					memset(&surfRes, 0, sizeof(cudaResourceDesc));
					surfRes.resType = cudaResourceTypeArray;
					surfRes.res.array.array = levelTo;
				}

				checkCudaErrors(cudaCreateSurfaceObject(&surfOutput, &surfRes));
			}

			// run mipmap kernel
			dim3 block(16, 16, 1);
			dim3 grid(
				(width + block.x - 1) / block.x,
				(height + block.y - 1) / block.y, 1);

			genMipmap << <grid, block >> > (
				surfOutput,
				texInput,
				width, height);

			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaGetLastError());

			checkCudaErrors(cudaDestroySurfaceObject(surfOutput));

			checkCudaErrors(cudaDestroyTextureObject(texInput));

			level++;
		}
	}

	static inline int getMipMapLevels(int width, int height)
	{
		int sz = std::max(width, height);

		int levels = 0;

		while (sz)
		{
			sz /= 2;
			levels++;
		}

		return levels;
	}

	void CudaTexture::initAsMipmap(
		const aten::vec4* p,
		int width, int height,
		int level)
	{
		level = std::min(level, getMipMapLevels(width, height));

		cudaExtent size;
		{
			size.width = width;
			size.height = height;
			size.depth = 0;
		}

		cudaMipmappedArray_t mipmapArray;

		cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
		checkCudaErrors(cudaMallocMipmappedArray(&mipmapArray, &desc, size, level));

		// upload level 0.
		cudaArray_t level0;
		checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, mipmapArray, 0));

		void* data = const_cast<void*>((const void*)p);

		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr(data, size.width * sizeof(float4), size.width, size.height);
		copyParams.dstArray = level0;
		copyParams.extent = size;
		copyParams.extent.depth = 1;
		copyParams.kind = cudaMemcpyHostToDevice;
		checkCudaErrors(cudaMemcpy3D(&copyParams));

		// compute rest of mipmaps based on level 0.
		onGenMipmaps(mipmapArray, width, height, level);

		// Make Resource description:
		memset(&m_resDesc, 0, sizeof(m_resDesc));
		m_resDesc.resType = cudaResourceTypeMipmappedArray;
		m_resDesc.res.mipmap.mipmap = mipmapArray;

		m_isMipmap = true;
		m_mipmapLevel = level;
	}
}