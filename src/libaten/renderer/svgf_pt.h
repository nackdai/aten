#pragma once

#include "renderer/renderer.h"
#include "scene/scene.h"
#include "camera/camera.h"
#include "light/pointlight.h"
#include "renderer/film.h"

namespace aten
{
	class SVGFPathTracing : public Renderer {
	public:
		SVGFPathTracing() {}
		~SVGFPathTracing() {}

		virtual void render(
			Destination& dst,
			scene* scene,
			camera* camera) override;

		struct AOV {
			vec3 normal;
			float depth;
			int meshid;
			int mtrlid;
			vec4 moments;
			vec4 color;
		};

	protected:
		struct Path {
			vec3 contrib;
			vec3 throughput;
			real pdfb{ 1 };

			hitrecord rec;
			material* prevMtrl{ nullptr };

			ray ray;

			bool isTerminate{ false };

			int idx;

			Path()
			{
				contrib = vec3(0);
				throughput = vec3(1);
			}
		};

		Path radiance(
			int x, int y,
			int w, int h,
			sampler* sampler,
			const ray& inRay,
			camera* cam,
			CameraSampleResult& camsample,
			scene* scene);

		Path radiance(
			int x, int y,
			int w, int h,
			sampler* sampler,
			uint32_t maxDepth,
			const ray& inRay,
			camera* cam,
			CameraSampleResult& camsample,
			scene* scene);

		bool shade(
			sampler* sampler,
			scene* scene,
			camera* cam,
			CameraSampleResult& camsample,
			int depth,
			Path& path);

		void shadeMiss(
			scene* scene,
			int depth,
			Path& path);

		void temporalReprojection(
			Destination& dst,
			camera* camera,
			int width, int height);

		void temporalReprojection(
			Destination& dst,
			camera* camera,
			int ix, int iy,
			int width, int height);

		const AOV& sampleAov(
			int ix, int iy,
			int width, int height);

		void estimateVariance(
			Destination& dst,
			int width, int height);

		void estimateVariance(
			Destination& dst,
			int ix, int iy,
			int width, int height);

		std::vector<AOV>& getCurAov()
		{
			return m_aov[m_curPos];
		}
		std::vector<AOV>& getPrevAov()
		{
			return m_aov[1 - m_curPos];
		}

	protected:
		uint32_t m_maxDepth{ 1 };
		uint32_t m_rrDepth{ 1 };

		int m_curPos{ 0 };

		std::vector<AOV> m_aov[2];

		aten::mat4 m_mtxV2C;
		aten::mat4 m_mtxC2V;
		aten::mat4 m_mtxPrevV2C;

		bool m_isFirstRender{ true };
	};
}
