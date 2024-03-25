#pragma once

#include <vector>
#include "defs.h"

namespace idaten {
    class CudaMemory {
    public:
        CudaMemory() {}

        CudaMemory(size_t bytes);
        CudaMemory(const void* p, size_t bytes);

        virtual ~CudaMemory();

    public:
        void resize(size_t bytes);

        const void* data() const
        {
            return m_device;
        }
        void* data()
        {
            return m_device;
        }

        size_t bytes() const
        {
            return m_bytes;
        }

        size_t writeFromHostToDeviceByBytes(const void* p, size_t sizeBytes, size_t offsetBytes = 0);
        size_t readFromDeviceToHostByBytes(void* p, size_t bytes);
        size_t readFromDeviceToHostByBytesWithOffset(void* p, size_t bytes, size_t offset_bytes);

        operator void*()
        {
            return m_device;
        }

        void free();

        bool empty() const
        {
            return (m_device == nullptr);
        }

        static size_t getHeapSize();

    private:
        uint8_t* m_device{ nullptr };
        size_t m_bytes{ 0 };
    };

    template <class _T>
    class TypedCudaMemory : public CudaMemory {
    public:
        TypedCudaMemory() {}

        TypedCudaMemory(size_t num)
            : CudaMemory(sizeof(_T) * num)
        {
            m_num = num;
        }
        TypedCudaMemory(const _T* p, size_t num)
            : CudaMemory(p, sizeof(_T) * num)
        {
            m_num = num;
        }

        virtual ~TypedCudaMemory() {}

        using value_type = _T;

    public:
        void resize(size_t num)
        {
            CudaMemory::resize(sizeof(_T) * num);
            m_num = num;
        }

        size_t writeFromHostToDeviceByNum(const _T* p, size_t num)
        {
            auto ret = CudaMemory::writeFromHostToDeviceByBytes(p, sizeof(_T) * num);
            return ret;
        }

        size_t readFromDeviceToHostByNum(void* p, size_t num = 0)
        {
            return CudaMemory::readFromDeviceToHostByBytes(p, sizeof(_T) * num);
        }

        size_t num() const
        {
            return m_num;
        }
        size_t size() const
        {
            return static_cast<size_t>(m_num);
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

        size_t readFromDeviceToHost(std::vector<_T>& host)
        {
            host.resize(m_num);
            return readFromDeviceToHostByNum(&host[0], m_num);
        }

        size_t readFromDeviceToHostByNumWithOffset(_T* host, size_t num, size_t offset_num)
        {
            auto bytes = sizeof(_T) * num;
            auto offset_bytes = sizeof(_T) * offset_num;
            return CudaMemory::readFromDeviceToHostByBytesWithOffset(&host[0], bytes, offset_bytes);
        }

    private:
        size_t m_num{ 0 };
    };
}
