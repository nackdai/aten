#pragma once

#include <vector>
#include "defs.h"

namespace idaten {
    class CudaMemory {
    public:
        CudaMemory() {}

        CudaMemory(uint32_t bytes);
        CudaMemory(const void* p, uint32_t bytes);

        virtual ~CudaMemory();

    public:
        void resize(uint32_t bytes);

        const void* data() const
        {
            return m_device;
        }
        void* data()
        {
            return m_device;
        }

        uint32_t bytes() const
        {
            return m_bytes;
        }

        uint32_t writeFromHostToDeviceByBytes(const void* p, uint32_t sizeBytes, uint32_t offsetBytes = 0);
        uint32_t readFromDeviceToHostByBytes(void* p, uint32_t bytes);
        uint32_t readFromDeviceToHostByBytesWithOffset(void* p, uint32_t bytes, uint32_t offset_bytes);

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
        uint8_t* m_device{ nullptr };
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

        using value_type = _T;

    public:
        void resize(uint32_t num)
        {
            CudaMemory::resize(sizeof(_T) * num);
            m_num = num;
        }

        uint32_t writeFromHostToDeviceByNum(const _T* p, uint32_t num)
        {
            auto ret = CudaMemory::writeFromHostToDeviceByBytes(p, sizeof(_T) * num);
            return ret;
        }

        uint32_t readFromDeviceToHostByNum(void* p, uint32_t num = 0)
        {
            return CudaMemory::readFromDeviceToHostByBytes(p, sizeof(_T) * num);
        }

        uint32_t num() const
        {
            return m_num;
        }

        const _T* data() const
        {
            return (const _T*)CudaMemory::data();
        }
        _T* data()
        {
            return (_T*)CudaMemory::data();
        }

        uint32_t stride()
        {
            return (uint32_t)sizeof(_T);
        }

        uint32_t readFromDeviceToHost(std::vector<_T>& host)
        {
            host.resize(m_num);
            return readFromDeviceToHostByNum(&host[0], m_num);
        }

        uint32_t readFromDeviceToHostByNumWithOffset(_T* host, uint32_t num, uint32_t offset_num)
        {
            auto bytes = sizeof(_T) * num;
            auto offset_bytes = sizeof(_T) * offset_num;
            return CudaMemory::readFromDeviceToHostByBytesWithOffset(&host[0], bytes, offset_bytes);
        }

    private:
        uint32_t m_num{ 0 };
    };
}
