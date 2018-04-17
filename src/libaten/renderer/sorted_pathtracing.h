#pragma once

#include "renderer/pathtracing.h"
#include "math/vec4.h"

namespace aten
{
	class SortedPathTracing : public PathTracing {
	public:
		SortedPathTracing() {}
		~SortedPathTracing() {}

		virtual void render(
			Destination& dst,
			scene* scene,
			camera* camera) override final;

	private:
		struct Path : public PathTracing::Path {
			CameraSampleResult camsample;
			real camSensitivity;

			uint32_t x, y;
			aten::sampler* sampler{ nullptr };

			vec3 lightcontrib;
			vec3 lightPos;
			Light* targetLight;
			
			struct {
				uint32_t isHit		: 1;
				uint32_t isAlive	: 1;
				uint32_t needWrite	: 1;
			};

			Path()
			{
				isHit = false;
				isAlive = true;
				needWrite = true;
			}
		};

		struct ShadowRay : aten::ray {
			ShadowRay() : ray()
			{
				isActive = true;
			}
			ShadowRay(const vec3& o, const vec3& d) : ray(o, d)
			{
				isActive = true;
			}

			struct {
				uint32_t isActive : 1;
			};
		};

		void makePaths(
			int width, int height,
			int sample,
			Path* paths,
			ray* rays,
			camera* camera);

		void hitPaths(
			Path* paths,
			const ray* rays,
			int numPath,
			scene* scene);

		int compactionPaths(
			Path* paths,
			int numPath,
			uint32_t* hitIds);

		void shadeMiss(
			scene* scene,
			int depth,
			Path* paths,
			int numPath,
			vec4* dst);

		void shade(
			uint32_t sample,
			uint32_t depth,
			Path* paths,
			ray* rays,
			ShadowRay* shadowRays,
			uint32_t* hitIds,
			int numHit,
			camera* cam,
			scene* scene);

		void hitShadowRays(
			const Path* paths,
			ShadowRay* shadowrays,
			int numRay,
			scene* scene);

		void evalExplicitLight(
			Path* paths,
			const ShadowRay* shadowRays,
			uint32_t* hitIds,
			int numHit);

		void gather(
			Path* paths,
			int numPath,
			vec4* dst);

	private:
		uint32_t m_maxDepth{ 1 };

		// Depth to compute russinan roulette.
		uint32_t m_rrDepth{ 1 };

		uint32_t m_samples{ 1 };

		std::vector<vec4> m_tmpbuffer;
		int m_width{ 0 };
		int m_height{ 0 };
	};
}
