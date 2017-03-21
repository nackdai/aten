#pragma once

#include "types.h"
#include "math/vec3.h"
#include "math/vec4.h"
#include "renderer/background.h"

namespace aten
{
	struct Destination {
		int width{ 0 };
		int height{ 0 };
		uint32_t maxDepth{ 1 };
		uint32_t russianRouletteDepth{ 1 };
		uint32_t startDepth{ 0 };
		uint32_t sample{ 1 };
		uint32_t mutation{ 1 };
		uint32_t mltNum{ 1 };
		vec4* buffer{ nullptr };
		vec4* variance{ nullptr };

		struct {
			vec4* nml_depth{ nullptr };		///< Normal and Depth / rgb : normal, a : depth
			vec4* albedo_vis{ nullptr };	///< Albedo and Visibility / rgb : albedo, a : visibility
			real depthMax{ 1 };
			bool needNormalize{ true };
		} geominfo;
	};

	class scene;
	class camera;

	class Renderer {
	protected:
		Renderer() {}
		virtual ~Renderer() {}

	public:
		virtual void render(
			Destination& dst,
			scene* scene,
			camera* camera) = 0;

		void setBG(background* bg)
		{
			m_bg = bg;
		}

	protected:
		virtual vec3 sampleBG(const ray& inRay) const
		{
			if (m_bg) {
				return m_bg->sample(inRay);
			}
			return std::move(vec3());
		}

		bool hasBG() const
		{
			return (m_bg != nullptr);
		}

		background* bg()
		{
			return m_bg;
		}

	private:
		background* m_bg{ nullptr };
	};
}
