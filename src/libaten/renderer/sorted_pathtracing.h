#pragma once

#include "renderer/pathtracing.h"

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

			sampler* sampler{ nullptr };
			
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
			scene* scene,
			int depth,
			Path* paths,
			int numPath,
			vec4* dst);

		void shade(
			uint32_t sample,
			uint32_t depth,
			Path* paths,
			uint32_t* hitIds,
			int numHit,
			camera* cam,
			scene* scene);

		void gather(
			Path* paths,
			int numPath,
			vec4* dst);

	private:
		uint32_t m_maxDepth{ 1 };

		// Depth to compute russinan roulette.
		uint32_t m_rrDepth{ 1 };

		uint32_t m_samples{ 1 };

		int m_width{ 0 };
		int m_height{ 0 };
	};
}
