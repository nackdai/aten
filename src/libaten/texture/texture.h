#pragma once

#include <vector>
#include "defs.h"
#include "types.h"
#include "math/vec3.h"

namespace aten {
	class texture {
	public:
		texture() {}
		texture(uint32_t width, uint32_t height, uint32_t channels)
		{
			m_width = width;
			m_height = height;
			m_channels = channels;

			m_pitch = width * channels;
			m_size = height * m_pitch;

			m_colors.resize(width * height * channels);
		}

		~texture() {}

	public:
		AT_DEVICE_API vec3 at(real u, real v) const
		{
			uint32_t x = (uint32_t)(aten::cmpMin(u, real(1)) * (m_width - 1));
			uint32_t y = (uint32_t)(aten::cmpMin(v, real(1)) * (m_height - 1));

			uint32_t pos = y * m_pitch + x * m_channels;

			const real* clr = &m_colors[pos];

			vec3 ret;

			switch (m_channels) {
			case 3:
				ret[2] = clr[2];
			case 2:
				ret[1] = clr[1];
			case 1:
				ret[0] = clr[0];
				break;
			}

			return ret;
		}

		real& operator[](uint32_t pos)
		{
			pos = std::min(pos, m_size - 1);
			return m_colors[pos];
		}

		real& operator()(uint32_t x, uint32_t y, uint32_t c)
		{
			x = std::min(x, m_width - 1);
			y = std::min(y, m_height - 1);
			c = std::min(c, m_channels - 1);

			auto pos = y * m_pitch + x * c;

			return m_colors[pos];
		}

		real* colors()
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

		uint32_t channels() const
		{
			return m_channels;
		}

	private:
		uint32_t m_width{ 0 };
		uint32_t m_height{ 0 };
		uint32_t m_channels{ 0 };

		uint32_t m_pitch{ 0 };
		uint32_t m_size{ 0 };

		std::vector<real> m_colors;
	};
}