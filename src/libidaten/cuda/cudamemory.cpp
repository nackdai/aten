#include <atomic>
#include "cuda/cudamemory.h"
#include "cuda/cudautil.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

	uint32_t CudaMemory::read(void* p, uint32_t size)
	{
		if (size == 0) {
			size = m_bytes;
		}

		if (size > m_bytes) {
			AT_ASSERT(false);
			return 0;
		}

		checkCudaErrors(cudaMemcpyAsync(p, m_device, size, cudaMemcpyDeviceToHost));

		return size;
	}

	void CudaMemory::free()
	{
		if (m_device) {
			checkCudaErrors(cudaFree(m_device));

			g_heapsize -= m_bytes;
		}
		m_bytes = 0;
	}

	uint32_t CudaMemory::getHeapSize()
	{
		return g_heapsize.load();
	}
}