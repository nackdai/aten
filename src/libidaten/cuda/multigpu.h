#pragma once

#include <cuda.h>
#include "cuda/cudautil.h"
#include "kernel/renderer.h"

#define ENABLE_MULTI_GPU_EMULATE

namespace idaten
{
	template <class RENDERER>
	class GpuProxy {
	public:
		GpuProxy() {}
		~GpuProxy() {}

	public:
		void init(int deviceId)
		{
#ifdef ENABLE_MULTI_GPU_EMULATE
			CUdevice device = 0;
			checkCudaErrors(cuDeviceGet(&device, deviceId));
			checkCudaErrors(cuCtxCreate(&m_context, 0, device));
#else
			checkCudaErrors(cudaSetDevice(deviceId));
			checkCudaErrors(cudaStreamCreate(&m_stream));
#endif

			m_deviceId = deviceId;
		}

		void setCurrent()
		{
#ifdef ENABLE_MULTI_GPU_EMULATE
			AT_ASSERT(m_context > 0);
			checkCudaErrors(cuCtxSetCurrent(m_context));
#else
			AT_ASSERT(m_deviceId >= 0);
			checkCudaErrors(cudaSetDevice(m_deviceId));
#endif
		}

		void render(
			const TileDomain& tileDomain,
			int maxSamples,
			int maxBounce)
		{
			setCurrent();
			m_renderer.render(tileDomain, maxSamples, maxBounce);
		}

		void shutdown()
		{
			setCurrent();

			// TODO
			// Shutdown renderer.

#ifdef ENABLE_MULTI_GPU_EMULATE
			checkCudaErrors(cuCtxDestroy(m_context));
#else
			checkCudaErrors(cudaStreamSynchronize(m_stream));
#endif
		}

		RENDERER& getRenderer()
		{
			return m_renderer;
		}

	private:
		int m_deviceId{ -1 };
		CUcontext m_context{ 0 };
		cudaStream_t m_stream{ 0 };

		RENDERER m_renderer;
	};
}