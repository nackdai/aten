#pragma once

#include <array>
#include <type_traits>

#include "defs.h"

namespace aten {
    template <class ElementType>
    class span {
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

#if 0
        using iterator = implementation - defined;
        using const_iterator = implementation - defined;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;
#endif

        // [span.cons], span constructors, copy, assignment, and destructor
        constexpr span() noexcept = default;
        constexpr span(pointer ptr, index_type count) : data_(ptr), size_(count) {}
        constexpr span(pointer firstElem, pointer lastElem) : span(firstElem, lastElem - firstElem) {}

        constexpr span(std::nullptr_t) {};

        template <size_t N>
        constexpr span(element_type(&arr)[N]) noexcept : span(arr, N) {}

        template <size_t N>
        constexpr span(std::array<element_type, N>& arr) noexcept : span(arr.data(), N) {}

        constexpr span(std::vector<element_type>& v) noexcept : span(v.data(), v.size()) {}

        constexpr span(const span& other) noexcept : span(other.data_, other.size_) {}

        ~span() noexcept = default;

        constexpr span& operator=(const span& other) noexcept
        {
            data_ = other.data_;
            size_ = other.size_;
            return *this;
        }

#if 0
        // [span.sub], span subviews
        template <size_t Count>
        constexpr span<element_type, Count> first() const
        template <size_t Count>
        constexpr span<element_type, Count> last() const;
        template <size_t Offset, size_t Count = dynamic_extent>
        constexpr span<element_type, see below> subspan() const;

        constexpr span<element_type, dynamic_extent> first(index_type count) const;
        constexpr span<element_type, dynamic_extent> last(index_type count) const;
        constexpr span<element_type, dynamic_extent> subspan(index_type offset, index_type count = dynamic_extent) const;
#endif
        constexpr span<element_type> first(index_type count) const
        {
            AT_ASSERT(count <= size());
            if (count > size()) {
                return span();
            }
            else {
                return { data_, count };
            }
        }
        constexpr span<element_type> last(index_type count) const
        {
            AT_ASSERT(count <= size());
            if (count > size()) {
                return span();
            }
            else {
                return { data_ + (size() - count), count };
            }
        }
        constexpr span<element_type> subspan(index_type offset, index_type count) const
        {
            AT_ASSERT(offset <= size() && offset + count <= size());
            if (offset > size() || offset + count > size()) {
                return span();
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
        constexpr reference operator[](index_type idx) const
        {
            AT_ASSERT(idx < size_);
            return data_[idx];
        }
        constexpr reference front() const
        {
            AT_ASSERT(!empty());
            return data_[0];
        }
        constexpr reference back() const
        {
            AT_ASSERT(!empty());
            return data_[size_ - 1];
        }
        constexpr pointer data() const noexcept
        {
            return data_;
        }

#if 0
        // [span.iterators], span iterator support
        constexpr iterator begin() const noexcept;
        constexpr iterator end() const noexcept;
        constexpr const_iterator cbegin() const noexcept;
        constexpr const_iterator cend() const noexcept;
        constexpr reverse_iterator rbegin() const noexcept;
        constexpr reverse_iterator rend() const noexcept;
        constexpr const_reverse_iterator crbegin() const noexcept;
        constexpr const_reverse_iterator crend() const noexcept;
#endif

    private:
        pointer data_{ nullptr };
        index_type size_{ 0 };
    };

}
