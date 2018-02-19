#pragma once

#include "types.h"

namespace aten {
	class Key {
	private:
		Key();
		~Key();

	public:
		// 新しいキー値を生成する
		static uint32_t gen(const char* str)
		{
			return genValue<uint32_t>(str);
		}
		static uint64_t gen64(const char* str)
		{
			return genValue<uint64_t>(str);
		}

		static uint32_t gen(uint32_t* p, uint32_t num)
		{
			return genValue<uint32_t, uint32_t>(p, num);
		}

		static uint64_t gen64(uint32_t* p, uint32_t num)
		{
			return genValue<uint64_t, uint32_t>(p, num);
		}

	private:
		template <typename _T>
		static _T genValue(const char* str)
		{
			if (str == nullptr) {
				AT_ASSERT(false);
				return 0;
			}

			_T ret = *str;
			_T k = 1;

			for (uint32_t i = 0; ; ++i) {
				if (str[i] == '\0') {
					break;
				}
				ret += (str[i] * (k *= 31L));
			}

			return ret;
		}

		template <typename _T, typename _U>
		static _T genValue(_U* p, uint32_t num)
		{
			if (p == nullptr) {
				AT_ASSERT(false);
				return 0;
			}

			uint8_t* pp = reinterpret_cast<uint8_t*>(p);
			num *= sizeof(_U);

			_T ret = *pp;
			_T k = 1;

			for (uint32_t i = 0; i < num; ++i) {
				ret += (pp[i] * (k *= 31L));
			}

			return ret;
		}
	};
}