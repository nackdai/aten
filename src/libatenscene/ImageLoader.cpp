#include "OpenImageIO/imageio.h"
#include "ImageLoader.h"
#include "AssetManager.h"
#include "texture/texture.h"
#include "utility.h"
#include <map>

namespace aten {
	static std::string g_base;

	void ImageLoader::setBasePath(const std::string& base)
	{
		g_base = removeTailPathSeparator(base);
	}

	template <typename TYPE>
	void read(
		OIIO::ImageInput* input,
		texture* tex, 
		OIIO::TypeDesc oioType,
		uint32_t srcChannels,
		real normalize)
	{
		const auto chn = tex->channels();

		int width = tex->width();
		int height = tex->height();

		auto size = width * height * srcChannels;

		// Resize storage
		std::vector<TYPE> texturedata(size);

		// Read data to storage
		input->read_image(oioType, &texturedata[0], sizeof(TYPE) * srcChannels);

		static const real div = real(1) / real(255.0);

#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
		for (int y = height - 1; y >= 0; y--) {
			for (int x = 0; x < width; x++) {
				// TODO
				// Invert y coordinate. Why?
				auto src_i = y * width + x;
				auto dst_i = ((height - 1) - y) * width + x;

				TYPE* s = &texturedata[src_i * srcChannels];

				switch (chn) {
				case 3:
					(*tex)[dst_i * chn + 2] = s[2] * normalize;
				case 2:
					(*tex)[dst_i * chn + 1] = s[1] * normalize;
				case 1:
					(*tex)[dst_i * chn + 0] = s[0] * normalize;
					break;
				}
			}
		}
	}

	texture* ImageLoader::load(const std::string& path)
	{
		std::string pathname;
		std::string extname;
		std::string filename;

		getStringsFromPath(
			path,
			pathname,
			extname,
			filename);

		auto tex = load(filename, path);

		return tex;
	}

	texture* ImageLoader::load(const std::string& tag, const std::string& path)
	{
		OIIO_NAMESPACE_USING

		std::string fullpath = path;
		if (!g_base.empty()) {
			fullpath = g_base + "/" + fullpath;
		}

		texture* tex = AssetManager::getTex(tag);

		if (tex) {
			AT_PRINTF("There is same tag texture. [%s]\n", tag.c_str());
			return tex;
		}

		ImageInput* input = ImageInput::open(fullpath);

		if (!input) {
			AT_ASSERT(false);
			return nullptr;
		}

		ImageSpec const& spec = input->spec();

		auto width = spec.width;
		auto height = spec.height;

		AT_ASSERT(spec.depth == 1);

		// ３チャンネル（RGB）まで.
		tex = new texture(width, height, std::min(spec.nchannels, 3));
		const auto chn = tex->channels();

		if (spec.format == TypeDesc::UINT8) {
			static const real div = real(1) / real(255.0);

			read<uint8_t>(input, tex, spec.format, spec.nchannels, div);
		}
		else if (spec.format == TypeDesc::HALF) {
			// TODO
			AT_ASSERT(false);
		}
		else {
			read<float>(input, tex, spec.format, spec.nchannels, real(1));
		}

		// Close handle
		input->close();

		if (tex) {
			AssetManager::registerTex(tag, tex);
		}

		return tex;
	}
}