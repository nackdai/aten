#pragma once

#include <array>
#include <type_traits>

#include "defs.h"
#include "misc/span.h"

namespace aten {
    template <class ElementType>
    class const_span {
    public:
        // constants and types
        using element_type = ElementType;
        using value_type = std::remove_cv_t<ElementType>;
        using index_type = size_t;
        using difference_type = ptrdiff_t;
        using pointer = element_type*;
        using const_pointer = const element_type*;
        using reference = element_type&;
        using const_reference = const element_type&;

        // [span.cons], span constructors, copy, assignment, and destructor
        constexpr const_span() noexcept = default;
        constexpr const_span(const_pointer ptr, index_type count) : data_(ptr), size_(count) {}
        constexpr const_span(const_pointer firstElem, const_pointer lastElem) : const_span(firstElem, lastElem - firstElem) {}

        constexpr const_span(std::nullptr_t) {};

        template <size_t N>
        constexpr const_span(const element_type(&arr)[N]) noexcept : const_span(arr, N) {}

        template <size_t N>
        constexpr const_span(const std::array<value_type, N>& arr) noexcept : const_span(arr.data(), N) {}

        template <class Container>
        constexpr const_span(const Container & cont) : const_span(cont.data(), cont.size()) {}

        constexpr const_span(const const_span& other) noexcept : const_span(other.data_, other.size_) {}

        constexpr const_span(span<ElementType>& other) noexcept : const_span(other.data(), other.size()) {}

        ~const_span() noexcept = default;

        constexpr const_span& operator=(const const_span& other) noexcept
        {
            data_ = other.data_;
            size_ = other.size_;
            return *this;
        }

        constexpr const_span<element_type> first(index_type count) const
        {
            AT_ASSERT(count <= size());
            if (count > size()) {
                return const_span();
            }
            else {
                return { data_, count };
            }
        }
        constexpr const_span<element_type> last(index_type count) const
        {
            AT_ASSERT(count <= size());
            if (count > size()) {
                return const_span();
            }
            else {
                return { data_ + (size() - count), count };
            }
        }
        constexpr const_span<element_type> subspan(index_type offset, index_type count) const
        {
            AT_ASSERT(offset <= size() && offset + count <= size());
            if (offset > size() || offset + count > size()) {
                return const_span();
            }
            else {
                return { data_ + offset, count };
            }
        }

        // [span.obs], span observers
        constexpr index_type size() const noexcept
        {
            return size_;
        }
        constexpr index_type size_bytes() const noexcept
        {
            return sizeof(element_type) * size_;
        }
        constexpr bool empty() const noexcept
        {
            return (data_ == nullptr) || (size_ == 0);
        }

        // [span.elem], span element access
        constexpr const_reference operator[](index_type idx) const
        {
            AT_ASSERT(idx < size_);
            return data_[idx];
        }
        constexpr const_reference front() const
        {
            AT_ASSERT(!empty());
            return data_[0];
        }
        constexpr const_reference back() const
        {
            AT_ASSERT(!empty());
            return data_[size_ - 1];
        }
        constexpr const_pointer data() const noexcept
        {
            return data_;
        }

    private:
        const_pointer data_{ nullptr };
        index_type size_{ 0 };
    };
}
