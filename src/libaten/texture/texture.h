#pragma once

#include <vector>
#include "defs.h"
#include "types.h"
#include "math/vec3.h"

namespace aten {
	class texture {
	public:
		texture() {}
		texture(uint32_t width, uint32_t height)
		{
			m_width = width;
			m_height = height;
			m_colors.resize(width * height);
		}

		~texture() {}

	public:
		const vec3& at(real u, real v) const
		{
			uint32_t x = (uint32_t)(std::min(u, real(1)) * (m_width - 1));
			uint32_t y = (uint32_t)(std::min(v, real(1)) * (m_height - 1));

			uint32_t pos = y * m_width + x;

			const vec3& ret = m_colors[pos];

			return ret;
		}

		vec3* colors()
		{
			return &m_colors[0];
		}

		uint32_t width() const
		{
			return m_width;
		}

		uint32_t height() const
		{
			return m_height;
		}

	private:
		uint32_t m_width{ 0 };
		uint32_t m_height{ 0 };
		std::vector<vec3> m_colors;
	};
}