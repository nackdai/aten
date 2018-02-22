#pragma once

#include "defs.h"
#include "types.h"

#ifndef AT_COUNT_BIT
	#define AT_COUNT_BIT(n)    (sizeof(n) << 3)
#endif

namespace aten
{
	template <typename _T>
	class BitFlag {
	public:
		BitFlag() {}
		BitFlag(_T val) { m_flag = val; }
		~BitFlag() {}

		BitFlag(const BitFlag& rhs) { m_flag = rhs.m_flag; }
		const BitFlag& operator=(const BitFlag& rhs)
		{
			m_flag = rhs.m_flag;
			return *this;
		}

	public:
		void set(_T val) { m_flag = val; }
		_T get() const { return m_flag; }

		void clear() { m_flag = 0; }

		bool isOn(_T mask) const { return ((m_flag & mask) > 0); }
		bool isOff(_T mask) const { return !isOn(mask); }

		bool isOnByBitShift(uint32_t bit) const
		{
			AT_ASSERT(bit < AT_COUNT_BIT(_T));
			return isOn(1 << bit);
		}
		bool isOffByBitShift(uint32_t bit) const { return !isOnByBitShift(bit); }

		void onFlag(_T mask) const { m_flag |= mask; }
		void offFlag(_T mask) const { m_flag &= ~mask; }

		void onFlagByBitShift(uint32_t bit) const
		{
			AT_ASSERT(bit < AT_COUNT_BIT(_T));
			m_flag |= (1 << bit);
		}
		void offFlagByBitShift(uint32_t bit) const
		{
			AT_ASSERT(bit < AT_COUNT_BIT(_T));
			m_flag &= ~(1 << bit);
		}

	private:
		_T m_flag{ 0 };
	};

	using Bit32Flag = BitFlag<uint32_t>;
	using Bit16Flag = BitFlag<uint16_t>;
	using Bit8Flag = BitFlag<uint8_t>;
}
