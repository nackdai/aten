#pragma once

#include "renderer/renderer.h"
#include "scene/scene.h"
#include "camera/camera.h"
#include "sampler/sampler.h"

namespace aten
{
	class BDPT : public Renderer {
	public:
		BDPT() {}
		~BDPT() {}

	public:
		virtual void render(
			Destination& dst,
			scene* scene,
			camera* camera) override;

	private:
		struct Vertex {
			vec3 pos;
			vec3 nml;

			hitable* obj{ nullptr };
			material* mtrl{ nullptr };

			Light* light{ nullptr };
			real sampleLightPdf{ real(0) };

			real u;
			real v;

			int x{ -1 };
			int y{ -1 };

			Vertex() {}

			Vertex(
				vec3 p,
				vec3 n,
				hitable* _obj,
				material* _mtrl,
				real _u, real _v,
				int _x = -1, int _y = -1)
				: pos(p), nml(n), obj(_obj), mtrl(_mtrl), u(_u), v(_v), x(_x), y(_y)
			{}

			Vertex(
				vec3 p,
				vec3 n,
				hitable* _obj,
				material* _mtrl,
				Light* _light,
				real lightPdf,
				real _u, real _v)
				: pos(p), nml(n), obj(_obj), mtrl(_mtrl), light(_light), sampleLightPdf(lightPdf), u(_u), v(_v)
			{}
		};

		struct Result {
			vec3 contrib;

			int x{ -1 };
			int y{ -1 };

			CameraSampleResult camsample;

			bool isStartFromPixel;

			Result() {}
			Result(const vec3& c, int _x, int _y, bool _isStartFromPixel)
				: contrib(c), x(_x), y(_y), isStartFromPixel(_isStartFromPixel)
			{}
		};

		vec3 trace(
			bool isEyePath,
			std::vector<Vertex>& vs,
			uint32_t maxDepth,
			const ray& r,
			sampler* sampler,
			scene* scene,
			camera* camera);

		Result genEyePath(
			std::vector<Vertex>& vs,
			int x, int y,
			sampler* sampler,
			scene* scene,
			camera* camera);

		Result genLightPath(
			std::vector<Vertex>& vs,
			Light* light,
			sampler* sampler,
			scene* scene,
			camera* camera);

		bool isConnectable(
			scene* scene,
			camera* cam,
			const std::vector<Vertex>& eyepath,
			const int numEye,
			const std::vector<Vertex>& lightpath,
			const int numLight,
			int& px, int& py);

		vec3 computeThroughput(
			const std::vector<Vertex>& vs,
			const CameraSampleResult& camsample,
			camera* camera);

		real BDPT::computePDF(
			const std::vector<Vertex>& vs,
			const CameraSampleResult& camsample,
			camera* camera,
			const int pathLenght,
			const int specNumEye = -1,
			const int specNumLight = -1);

		real computeMISWeight(
			const std::vector<Vertex>& vs,
			const CameraSampleResult& camsample,
			camera* camera,
			const int numEyeVertices,
			const int numLightVertices,
			const int pathLength);

		void combinePath(
			std::vector<Result>& result,
			const std::vector<Vertex>& eyepath,
			const std::vector<Vertex>& lightpath,
			scene* scene,
			const CameraSampleResult& camsample,
			camera* camera);

	private:
		uint32_t m_maxDepth{ 1 };

		int m_width;
		int m_height;
	};
}
