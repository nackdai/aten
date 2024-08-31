#pragma once

#include <array>
#include <optional>

namespace aten
{
    template <class T, size_t N>
    class stack {
    public:
        stack() = default;
        ~stack() = default;

        stack(const stack&) = default;
        stack(stack&&) = default;
        stack& operator=(const stack&) = default;
        stack& operator=(stack&&) = default;

        AT_DEVICE_API bool empty() const
        {
            return size_ == 0;
        }

        AT_DEVICE_API size_t size() const
        {
            return size_;
        }

        T& top()
        {
            AT_ASSERT(size_ > 0);
            return queue_.front();
        }
        AT_DEVICE_API const T& top() const
        {
            AT_ASSERT(size_ > 0);
            return queue_.front();
        }

        T& back()
        {
            AT_ASSERT(size_ > 0);
            return queue_[size_ - 1];
        }
        const T& back() const
        {
            AT_ASSERT(size_ > 0);
            return queue_[size_ - 1];
        }

        AT_DEVICE_API void push(const T& item)
        {
            AT_ASSERT(size_ < N);
            queue_[size_] = item;
            size_ += 1;
        }
        void push(T&& item)
        {
            AT_ASSERT(size_ < N);
            queue_[size_] = std::move(item);
            size_ += 1;
        }

        AT_DEVICE_API  T& pop()
        {
            AT_ASSERT(size_ > 0);
            T& ret = queue_[size_ - 1];
            size_ -= 1;
            return ret;
        }

        std::optional<T> safe_pop()
        {
            if (size_ > 0) {
                T& ret = queue_[size_ - 1];
                size_ -= 1;
                return ret;
            }
            return std::nullopt;
        }

        AT_DEVICE_API void clear()
        {
            size_ = 0;
        }

    private:
        size_t size_{ 0 };
        std::array<T, N> queue_;
    };
}
