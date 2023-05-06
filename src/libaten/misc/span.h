#pragma once

#include <vector>

#include "defs.h"

namespace aten {
    template <typename T>
    class span {
    public:
        using element_type = T;
        using value_type = std::remove_cv_t<T>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using const_pointer = const T*;
        using reference = T&;
        using const_reference = const T&;
        using iterator = pointer;
        using const_iterator = const_pointer;

        AT_DEVICE_API span() noexcept = default;
        AT_DEVICE_API span(pointer data, size_type size) noexcept : data_(data), size_(size) {}

        span(const std::vector<T>& vec) noexcept : data_(vec.data()), size_(vec.size()) {}

        AT_DEVICE_API ~span() = default;

        span(const span& rhs) noexcept : data_(rhs.data_), size_(rhs.size_) {}
        span& operator=(const Span& rhs) noexcept
        {
            data_ = rhs.data_;
            size_ = rhs.size_;
            return *this;
        }

        Span(Span&&) = delete;
        Span& operator=(Span&&) = delete;

        AT_DEVICE_API pointer data() const noexcept { return data_; }
        AT_DEVICE_API size_type size() const noexcept { return size_; }
        AT_DEVICE_API bool empty() const noexcept { return size_ == 0; }
        AT_DEVICE_API reference operator[](size_type index) const noexcept { return data_[index]; }
        AT_DEVICE_API reference front() const noexcept { return data_[0]; }
        AT_DEVICE_API reference back() const noexcept { return data_[size_ - 1]; }
        AT_DEVICE_API iterator begin() const noexcept { return data_; }
        AT_DEVICE_API iterator end() const noexcept { return data_ + size_; }
        AT_DEVICE_API const_iterator cbegin() const noexcept { return data_; }
        AT_DEVICE_API const_iterator cend() const noexcept { return data_ + size_; }
        AT_DEVICE_API span subspan(size_type offset, size_type count = size_type(-1)) const noexcept
        {
            assert(offset <= size_);
            return span(data_ + offset, std::min(count, size_ - offset));
        }

    private:
        pointer data_{ nullptr };
        size_type size_{ 0 };
    };
}
