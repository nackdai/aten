#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "aten4idaten.h"

namespace idaten {
	class CudaMemory : public aten::IStream {
	public:
		CudaMemory() {}

		CudaMemory(uint32_t bytes);
		CudaMemory(const void* p, uint32_t bytes);

		virtual ~CudaMemory();

	public:
		void init(uint32_t bytes);

		const void* ptr() const
		{
			return m_device;
		}
		void* ptr()
		{
			return m_device;
		}
		
		uint32_t bytes() const
		{
			return m_bytes;
		}

		virtual __host__ uint32_t write(const void* p, uint32_t size) override final;
		virtual __host__ uint32_t read(void* p, uint32_t size) override final;

		operator void*()
		{
			return m_device;
		}

		void reset();

		void free();

		static uint32_t getHeapSize();

	private:
		void* m_device{ nullptr };
		uint32_t m_bytes{ 0 };
		uint32_t m_pos{ 0 };
	};

	template <typename _T>
	class TypedCudaMemory : public CudaMemory {
	public:
		TypedCudaMemory() {}

		TypedCudaMemory(uint32_t num)
			: CudaMemory(sizeof(_T) * num)
		{
			m_num = num;
		}
		TypedCudaMemory(const _T* p, uint32_t num)
			: CudaMemory(p, sizeof(_T) * num)
		{
			m_num = num;
		}

		virtual ~TypedCudaMemory() {}

	public:
		void init(uint32_t num)
		{
			CudaMemory::init(sizeof(_T) * num);
			m_num = num;
		}

		__host__ uint32_t writeByNum(const _T* p, uint32_t num)
		{
			auto ret = CudaMemory::write(p, sizeof(_T) * num);
			if (ret > 0) {
				m_cur += num;
			}
			return ret;
		}

		__host__ uint32_t readByNum(void* p, uint32_t num = 0)
		{
			return CudaMemory::read(p, sizeof(_T) * num);
		}

		uint32_t maxNum() const
		{
			return m_num;
		}

		uint32_t num() const
		{
			return m_cur;
		}

		const _T* ptr() const
		{
			return (const _T*)CudaMemory::ptr();
		}
		_T* ptr()
		{
			return (_T*)CudaMemory::ptr();
		}

	private:
		uint32_t m_num{ 0 };
		uint32_t m_cur{ 0 };
	};
}