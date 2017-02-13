#include "OpenImageIO/imageio.h"
#include "texture/ImageLoader.h"
#include "texture/texture.h"

namespace aten {
	texture* ImageLoader::load(const std::string& path)
	{
		OIIO_NAMESPACE_USING

		ImageInput* input = ImageInput::open(path);

		if (!input) {
			AT_ASSERT(false);
			return nullptr;
		}

		ImageSpec const& spec = input->spec();

		auto width = spec.width;
		auto height = spec.height;

		AT_ASSERT(spec.depth == 1);

		texture* tex = new texture(width, height);
		auto dst = tex->colors();

		if (spec.format == TypeDesc::UINT8) {
			auto size = width * height * 4;

			// Resize storage
			std::vector<uint8_t> texturedata(size);

			// Read data to storage
			input->read_image(TypeDesc::UINT8, &texturedata[0], sizeof(uint8_t) * 4);

			static const real div = CONST_REAL(255.0);

			// Convert to vec3 and normalize.
			for (uint32_t i = 0; i < width * height; i++) {
				uint8_t* s = &texturedata[i * 4];
				vec3& d = dst[i];

				d.r = s[0] / div;
				d.g = s[1] / div;
				d.b = s[2] / div;
			}
		}
		else if (spec.format == TypeDesc::HALF) {
			// TODO
			AT_ASSERT(false);
		}
		else {
			auto size = width * height * 3;

			// Resize storage
			std::vector<float> texturedata(size);

			// Read data to storage
			input->read_image(TypeDesc::FLOAT, &texturedata[0], sizeof(float) * 3);

			// Convert to vec3.
			for (int y = height - 1; y >= 0; y--) {
				for (int x = 0; x < width; x++) {
					// TODO
					// Invert y coordinate. Why?
					auto src_i = y * width + x;
					auto dst_i = ((height - 1) - y) * width + x;

					float* s = &texturedata[src_i * 3];
					vec3& d = dst[dst_i];

					d.r = s[0];
					d.g = s[1];
					d.b = s[2];
				}
			}
		}

		// Close handle
		input->close();

		return tex;
	}
}