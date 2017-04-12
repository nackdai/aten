#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "aten.h"

namespace aten {
	class CudaMemory : public IStream {
	public:
		CudaMemory() {}

		CudaMemory(uint32_t bytes);
		CudaMemory(void* p, uint32_t bytes);

		virtual ~CudaMemory();

	public:
		void init(uint32_t bytes);

		void* ptr()
		{
			return m_device;
		}
		
		uint32_t bytes() const
		{
			return m_bytes;
		}

		virtual __host__ uint32_t write(void* p, uint32_t size) override final;
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
}