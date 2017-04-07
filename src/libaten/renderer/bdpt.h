#pragma once

#include "renderer/renderer.h"
#include "scene/scene.h"
#include "camera/camera.h"
#include "sampler/sampler.h"

namespace aten
{
	struct Vertex {
		vec3 pos;
		vec3 nml;

		hitable* obj;
		material* mtrl;

		real u;
		real v;

		Vertex() {}

		Vertex(
			vec3 p,
			vec3 n,
			hitable* _obj,
			material* _mtrl,
			real _u, real _v)
			: pos(p), nml(n), obj(_obj), mtrl(_mtrl), u(_u), v(_v)
		{
		}
	};

	class BDPT {
	private:
		BDPT() {}
		~BDPT() {}

	private:
		vec3 trace(
			bool isEyePath,
			std::vector<Vertex>& vs,
			uint32_t maxDepth,
			const ray& r,
			sampler* sampler,
			scene* scene) const;

		vec3 genEyePath(
			std::vector<Vertex>& vs,
			real u, real v,
			sampler* sampler,
			scene* scene,
			camera* cam) const;

		void genLightPath(
			std::vector<Vertex>& vs,
			Light* light,
			sampler* sampler,
			scene* scene) const;

		bool isConnectable(
			scene* scene,
			camera* cam,
			const std::vector<Vertex>& eyepath,
			const int numEye,
			const std::vector<Vertex>& lightpath,
			const int numLight) const;

		void BDPT::contribution(
			std::vector<vec3>& result,
			const std::vector<Vertex>& eyepath,
			const std::vector<Vertex>& lightpath,
			scene* scene,
			const CameraSampleResult& camsample,
			camera* camera) const;

		vec3 render(
			std::vector<Vertex>& vs,
			const CameraSampleResult& camsample,
			sampler* sampler,
			scene* scene,
			camera* camera);

	private:
		struct Path {
			vec3 contrib{ vec3(0) };
			vec3 throughput{ vec3(1) };
			real pdfb{ 1 };

			hitrecord rec;
			material* prevMtrl{ nullptr };

			ray ray;

			bool isTerminate{ false };
		};

		uint32_t m_maxDepth{ 1 };

		int m_width;
		int m_height;
	};
}
