#pragma once

#include "defs.h"

namespace idaten {
	class CudaMemory {
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

		uint32_t write(const void* p, uint32_t sizeBytes, uint32_t offsetBytes = 0);
		uint32_t read(void* p, uint32_t size);

		operator void*()
		{
			return m_device;
		}

		void free();

		bool empty() const
		{
			return (m_device == nullptr);
		}

		static uint32_t getHeapSize();

	private:
		void* m_device{ nullptr };
		uint32_t m_bytes{ 0 };
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

		uint32_t writeByNum(const _T* p, uint32_t num)
		{
			auto ret = CudaMemory::write(p, sizeof(_T) * num);
			return ret;
		}

		uint32_t readByNum(void* p, uint32_t num = 0)
		{
			return CudaMemory::read(p, sizeof(_T) * num);
		}

		uint32_t num() const
		{
			return m_num;
		}

		const _T* ptr() const
		{
			return (const _T*)CudaMemory::ptr();
		}
		_T* ptr()
		{
			return (_T*)CudaMemory::ptr();
		}

		uint32_t stride()
		{
			return (uint32_t)sizeof(_T);
		}

	private:
		uint32_t m_num{ 0 };
	};
}