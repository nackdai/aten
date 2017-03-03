#pragma once

#include "renderer/renderer.h"
#include "scene/scene.h"
#include "camera/camera.h"

namespace aten
{
	class PathTracing : public Renderer {
	public:
		PathTracing() {}
		~PathTracing() {}

		virtual void render(
			Destination& dst,
			scene* scene,
			camera* camera) override final;

	private:
		struct Path {
			vec3 contrib;
			bool isTerminate{ false };
		};

		Path radiance(
			sampler* sampler,
			const ray& inRay,
			camera* cam,
			CameraSampleResult& camsample,
			scene* scene);

		Path radiance(
			sampler* sampler,
			uint32_t maxDepth,
			const ray& inRay,
			camera* cam,
			CameraSampleResult& camsample,
			scene* scene);

	private:
		uint32_t m_maxDepth{ 1 };

		// Depth to compute russinan roulette.
		uint32_t m_rrDepth{ 1 };
	};

#if 1
	class SortedPathTracing : public Renderer {
	public:
		SortedPathTracing() {}
		~SortedPathTracing() {}

		virtual void render(
			Destination& dst,
			scene* scene,
			camera* camera) override final;

	private:
		struct Path {
			ray r;
			CameraSampleResult camsample;
			hitrecord rec;
			vec3 throughput{ vec3(1, 1, 1) };

			uint32_t x, y;

			sampler* sampler{ nullptr };
			
			struct {
				uint32_t isHit		: 1;
				uint32_t isAlive	: 1;
			};

			Path()
			{
				isHit = false;
				isAlive = false;
			}
		};

		void makePaths(
			int width, int height,
			int sample,
			Path* paths,
			camera* camera);

		void hitPaths(
			Path* paths,
			int numPath,
			scene* scene);

		int compactionPaths(
			Path* paths,
			int numPath,
			uint32_t* hitIds);

		void shadeMiss(
			Path* paths,
			int numPath,
			vec3* dst);

		void shade(
			uint32_t depth,
			Path* paths,
			uint32_t* hitIds,
			int numHit,
			camera* cam,
			scene* scene,
			vec3* dst);

	private:
		uint32_t m_maxDepth{ 1 };

		// Depth to compute russinan roulette.
		uint32_t m_rrDepth{ 1 };

		uint32_t m_samples{ 1 };
	};
#endif
}
