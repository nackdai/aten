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

	__host__ uint32_t CudaMemory::write(const void* p, uint32_t size)
	{
		if (!m_device) {
			init(size);
		}

		if (m_pos + size > m_bytes) {
			AT_ASSERT(false);
			return 0;
		}

		uint8_t* dst = (uint8_t*)m_device;
		dst += m_pos;

		checkCudaErrors(cudaMemcpy(dst, p, size, cudaMemcpyHostToDevice));

		m_pos += size;

		return size;
	}

	__host__ uint32_t CudaMemory::read(void* p, uint32_t size)
	{
		if (size == 0) {
			size = m_bytes;
		}

		if (size > m_bytes) {
			AT_ASSERT(false);
			return 0;
		}

		checkCudaErrors(cudaMemcpy(p, m_device, size, cudaMemcpyDeviceToHost));

		return size;
	}

	void CudaMemory::reset()
	{
		m_pos = 0;
	}

	void CudaMemory::free()
	{
		if (m_device) {
			checkCudaErrors(cudaFree(m_device));

			g_heapsize -= m_bytes;
		}
		m_pos = 0;
		m_bytes = 0;
	}

	uint32_t CudaMemory::getHeapSize()
	{
		return g_heapsize.load();
	}
}