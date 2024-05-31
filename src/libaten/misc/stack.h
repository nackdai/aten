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

        bool empty() const
        {
            return size_ == 0;
        }

        size_t size() const
        {
            return size_;
        }

        T& front()
        {
            AT_ASSERT(size_ > 0);
            return queue_.front();
        }
        const T& front() const
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

        void push(const T& item)
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

        T& pop()
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

    private:
        size_t size_{ 0 };
        std::array<T, N> queue_;
    };
}
