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
			checkCudaErrors(cuDeviceGet(&device, 0));
			checkCudaErrors(cuCtxCreate(&m_context, 0, device));
#else
			checkCudaErrors(cudaSetDevice(deviceId));
			checkCudaErrors(cudaStreamCreate(&m_stream));
#endif

			m_deviceId = deviceId;
		}

		void setPeerAccess(int peerAccessDeviceId)
		{
			AT_ASSERT(m_deviceId >= 0);

			// NOTE
			// https://stackoverflow.com/questions/22694518/what-is-the-difference-between-cudamemcpy-and-cudamemcpypeer-for-p2p-copy

#ifdef ENABLE_MULTI_GPU_EMULATE
			// Nothing is done...
#else
			if (m_deviceId != peerAccessDeviceId) {
				// Check for peer access between participating GPUs.
				int canAccessPeer = 0;
				checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeer, m_deviceId, peerAccessDeviceId));

				AT_ASSERT(canAccessPeer > 0);

				if (canAccessPeer > 0) {
					// Enable peer access between participating GPUs.
					setCurrent();
					checkCudaErrors(cudaDeviceEnablePeerAccess(peerAccessDeviceId, 0));
				}
			}
#endif
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

		void postRender(int width = 0, int height = 0)
		{
			setCurrent();
			m_renderer.postRender(width, height);
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

		void gather(GpuProxy& proxy)
		{
			m_renderer.copyFrom(proxy.m_renderer);
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