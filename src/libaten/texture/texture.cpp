#include "texture/texture.h"

namespace aten {
	std::vector<texture*> texture::g_textures;

	texture::texture()
	{
		m_id = (int)g_textures.size();
		g_textures.push_back(this);
	}

	texture::texture(uint32_t width, uint32_t height, uint32_t channels)
		: texture()
	{
		m_width = width;
		m_height = height;
		m_channels = channels;

		m_size = height * width;

		m_colors.resize(width * height);
	}

	texture::~texture()
	{
		auto found = std::find(g_textures.begin(), g_textures.end(), this);
		if (found != g_textures.end()) {
			g_textures.erase(found);
		}
	}

	const texture* texture::getTexture(int id)
	{
		AT_ASSERT(g_textures.size() > id);
		return g_textures[id];
	}

	const std::vector<texture*>& texture::getTextures()
	{
		return g_textures;
	}
}