#pragma once

#include "cuda/cudadefs.h"
#include "cuda/cudautil.h"
#include "kernel/renderer.h"

//#define ENABLE_MULTI_GPU_EMULATE

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

            m_renderer.setStream(m_stream);
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
            if (m_deviceId != peerAccessDeviceId)
            {
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

        void copy(GpuProxy& from)
        {
            setCurrent();

            m_renderer.copy(from.m_renderer, from.m_stream);
        }

        static void swapCopy(GpuProxy* proxies, int num)
        {
            // TODO
            AT_ASSERT(num == 4);

            checkCudaErrors(cudaStreamSynchronize(proxies[1].m_stream));
            checkCudaErrors(cudaStreamSynchronize(proxies[3].m_stream));

            {
                auto keepTileDomain = proxies[1].m_renderer.m_tileDomain;
                proxies[1].m_renderer.m_tileDomain.x = 0;
                proxies[1].m_renderer.m_tileDomain.y = proxies[1].m_renderer.m_tileDomain.h;

                proxies[0].copy(proxies[1]);

                proxies[1].m_renderer.m_tileDomain = keepTileDomain;
            }

            {
                auto keepTileDomain = proxies[3].m_renderer.m_tileDomain;
                proxies[3].m_renderer.m_tileDomain.x = 0;
                proxies[3].m_renderer.m_tileDomain.y = proxies[3].m_renderer.m_tileDomain.h;

                proxies[2].copy(proxies[3]);

                proxies[3].m_renderer.m_tileDomain = keepTileDomain;
            }

            {
                auto keepTileDomain_0 = proxies[0].m_renderer.m_tileDomain;
                auto keepTileDomain_1 = proxies[2].m_renderer.m_tileDomain;

                proxies[0].m_renderer.m_tileDomain.h *= 2;
                proxies[2].m_renderer.m_tileDomain.h *= 2;

                proxies[2].setCurrent();

                checkCudaErrors(cudaStreamSynchronize(proxies[3].m_stream));

                proxies[0].m_renderer.copy(proxies[2].m_renderer, proxies[2].m_stream);

                proxies[0].m_renderer.m_tileDomain = keepTileDomain_0;
                proxies[2].m_renderer.m_tileDomain = keepTileDomain_1;

                checkCudaErrors(cudaStreamSynchronize(proxies[2].m_stream));
                checkCudaErrors(cudaStreamSynchronize(proxies[0].m_stream));
            }
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
