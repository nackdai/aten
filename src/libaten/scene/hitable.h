#pragma once 

#include <vector>
#include "types.h"
#include "math/aabb.h"
#include "math/vec3.h"
#include "material/material.h"
#include "sampler/sampler.h"
#include "shape/shape.h"

namespace aten {
	class hitable;

	struct hitrecord {
		real t{ AT_MATH_INF };

		vec3 p;

		vec3 normal;

		// tangent coordinate.
		vec3 du;
		vec3 dv;

		// texture coordinate.
		real u{ real(0) };
		real v{ real(0) };

		real area{ real(1) };

		union {
			// TODO
#ifdef __AT_CUDA__
			void* obj{ nullptr };
#else
			hitable* obj{ nullptr };
#endif
			int objid;
		};

		union {
#ifdef __AT_CUDA__
			void* mtrl{ nullptr };
#else
			material* mtrl{ nullptr };
#endif
			int mtrlid;
		};

		union {
			// cube.
			struct {
				int face;
			};
			// triangle.
			struct {
				int idx[3];
				real a, b;	// barycentric
			};
		} param;
	};

	class hitable {
	public:
		hitable(const char* name = nullptr);
		virtual ~hitable() {}

	public:
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const = 0;

		virtual void evalHitResult(const ray& r, hitrecord& rec) const
		{
			AT_ASSERT(false);
		}

		virtual aabb getBoundingbox() const = 0;

		struct SamplePosNormalPdfResult {
			aten::vec3 pos;
			aten::vec3 nml;
			real area;

			real a;
			real b;
			int idx[3];
		};

		virtual void getSamplePosNormalArea(SamplePosNormalPdfResult* result, sampler* sampler) const
		{
			AT_ASSERT(false);
		}

		uint32_t id() const
		{
			return m_id;
		}

	private:
		uint32_t m_id{ 0 };
		const char* m_name;
	};
}
