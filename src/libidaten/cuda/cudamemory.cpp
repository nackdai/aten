#include <atomic>
#include "cuda/cudamemory.h"
#include "cuda/cudautil.h"

namespace idaten {
    static std::atomic<size_t> g_heapsize(0);

    CudaMemory::CudaMemory(size_t bytes)
    {
        resize(bytes);
    }

    CudaMemory::CudaMemory(const void* p, size_t bytes)
    {
        resize(bytes);
        writeFromHostToDeviceByBytes(p, bytes);
    }

    CudaMemory::~CudaMemory()
    {
        free();
    }

    void CudaMemory::resize(size_t bytes)
    {
        if (m_bytes != bytes) {
            free();

            checkCudaErrors(cudaMalloc((void**)&m_device, bytes));
            m_bytes = bytes;

            g_heapsize += bytes;
        }
    }

    size_t CudaMemory::writeFromHostToDeviceByBytes(const void* p, size_t sizeBytes, size_t offsetBytes/*= 0*/)
    {
        if (!m_device) {
            resize(sizeBytes);
        }

        if (sizeBytes > m_bytes) {
            AT_ASSERT(false);
            return 0;
        }

        uint8_t* dst = (uint8_t*)m_device;

        checkCudaErrors(cudaMemcpyAsync(dst + offsetBytes, p, sizeBytes, cudaMemcpyDefault));

        return sizeBytes;
    }

    size_t CudaMemory::readFromDeviceToHostByBytes(void* p, size_t bytes)
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

    size_t CudaMemory::readFromDeviceToHostByBytesWithOffset(void* p, size_t bytes, size_t offset_bytes)
    {
        if (bytes == 0) {
            bytes = m_bytes;
        }

        if (bytes + offset_bytes > m_bytes) {
            AT_ASSERT(false);
            return 0;
        }

        // NOTE
        // https://stackoverflow.com/questions/64989922/cuda-is-it-safe-to-copy-a-single-element-from-device-memory-by-array-offset
        checkCudaErrors(cudaMemcpy(p, m_device + offset_bytes, bytes, cudaMemcpyDeviceToHost));

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

    size_t CudaMemory::getHeapSize()
    {
        return g_heapsize.load();
    }
}
