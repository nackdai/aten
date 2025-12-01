#pragma once

#include <chrono>

#include "defs.h"
#include "types.h"

namespace AT_NAME {
    class timer {
    public:
        timer() {}
        ~timer() {}

    public:
        /**
         * @brief Begin to measure the elapsed time.
         */
        void begin()
        {
            start_ = std::chrono::system_clock::now();
            is_measuring_ = true;
        }

        /**
         * @brief End to measure the elapsed time.
         * @return Elapsed time as milliseconds.
         */
        int64_t end()
        {
            AT_ASSERT(is_measuring_);
            auto end = std::chrono::system_clock::now();
            is_measuring_ = false;

            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
            return elapsed_ms.count();
        }

        /**
         * @brief Get current milliseconds.
         * @return Milliseconds of the current time.
         */
        static int64_t GetCurrMilliseconds()
        {
            auto now = std::chrono::system_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
            return static_cast<int64_t>(ms.count());
        }

    private:
        bool is_measuring_{ false };
        std::chrono::system_clock::time_point start_;
    };
}
