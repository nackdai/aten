#include "texture/texture.h"

#include "visualizer/atengl.h"
#include "visualizer/shader.h"

#include <string>

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
		init(width, height, channels);
	}

	texture::~texture()
	{
		auto found = std::find(g_textures.begin(), g_textures.end(), this);
		if (found != g_textures.end()) {
			g_textures.erase(found);
		}

		releaseAsGLTexture();
	}

	void texture::init(uint32_t width, uint32_t height, uint32_t channels)
	{
		if (m_colors.empty()) {
			m_width = width;
			m_height = height;
			m_channels = channels;

			m_size = height * width;

			m_colors.resize(width * height);
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

	bool texture::initAsGLTexture()
	{
		AT_VRETURN(m_width > 0, false);
		AT_VRETURN(m_height > 0, false);
		AT_VRETURN(m_colors.size() > 0, false);

		CALL_GL_API(::glGenTextures(1, &m_gltex));
		AT_VRETURN(m_gltex > 0, false);

		CALL_GL_API(glBindTexture(GL_TEXTURE_2D, m_gltex));

		CALL_GL_API(glTexImage2D(
			GL_TEXTURE_2D,
			0,
			GL_RGBA32F,
			m_width, m_height,
			0,
			GL_RGBA,
			GL_FLOAT,
			&m_colors[0]));

		CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
		CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));

		CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP));
		CALL_GL_API(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP));

		CALL_GL_API(::glBindTexture(GL_TEXTURE_2D, 0));

		return true;
	}

	void texture::bindAsGLTexture(uint8_t stage, shader* shd) const
	{
		AT_ASSERT(m_gltex > 0);
		AT_ASSERT(shd);

		// NOTE
		// shader‚ÍƒoƒCƒ“ƒh‚³‚ê‚Ä‚¢‚é‚±‚Æ.

		std::string texuniform = std::string("s") + std::to_string(stage);
		auto handle = shd->getHandle(texuniform.c_str());
		AT_ASSERT(handle >= 0);
		CALL_GL_API(::glUniform1i(handle, stage));

		CALL_GL_API(::glActiveTexture(GL_TEXTURE0 + stage));

		CALL_GL_API(glBindTexture(GL_TEXTURE_2D, m_gltex));

		CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
		CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));

		CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP));
		CALL_GL_API(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP));
	}

	void texture::releaseAsGLTexture()
	{
		if (m_gltex > 0) {
			CALL_GL_API(::glDeleteTextures(1, &m_gltex));
			m_gltex = 0;
		}
	}
}