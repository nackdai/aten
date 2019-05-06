#include <atomic>
#include "cuda/cudamemory.h"
#include "cuda/cudautil.h"

namespace idaten {
    static std::atomic<uint32_t> g_heapsize(0);

    CudaMemory::CudaMemory(uint32_t bytes)
    {
        init(bytes);
    }

    CudaMemory::CudaMemory(const void* p, uint32_t bytes)
    {
        init(bytes);
        write(p, bytes);
    }

    CudaMemory::~CudaMemory()
    {
        free();
    }

    void CudaMemory::init(uint32_t bytes)
    {
        if (m_bytes == 0) {
            checkCudaErrors(cudaMalloc((void**)&m_device, bytes));
            m_bytes = bytes;

            g_heapsize += bytes;
        }
    }

    uint32_t CudaMemory::write(const void* p, uint32_t sizeBytes, uint32_t offsetBytes/*= 0*/)
    {
        if (!m_device) {
            init(sizeBytes);
        }

        if (sizeBytes > m_bytes) {
            AT_ASSERT(false);
            return 0;
        }

        uint8_t* dst = (uint8_t*)m_device;

        checkCudaErrors(cudaMemcpyAsync(dst + offsetBytes, p, sizeBytes, cudaMemcpyDefault));

        return sizeBytes;
    }

    uint32_t CudaMemory::read(void* p, uint32_t bytes)
    {
        if (bytes == 0) {
            bytes = m_bytes;
        }

        if (bytes > m_bytes) {
            AT_ASSERT(false);
            return 0;
        }

        checkCudaErrors(cudaMemcpy(p, m_device, bytes, cudaMemcpyDeviceToHost));

        return bytes;
    }

    void CudaMemory::free()
    {
        if (m_device) {
            checkCudaErrors(cudaFree(m_device));
            m_device = nullptr;

            g_heapsize -= m_bytes;
        }
        m_bytes = 0;
    }

    uint32_t CudaMemory::getHeapSize()
    {
        return g_heapsize.load();
    }
}