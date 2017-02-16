#pragma once

#include "types.h"
#include "scene/bvh.h"
#include "material/material.h"

namespace aten
{
	struct vertex {
		vec3 pos;
		vec3 nml;
		real uv[2];
	};

	struct face {
		uint32_t idx[3];
	};

	class shape : public hitable {
	public:
		shape() {}
		virtual ~shape() {}

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override final;

		virtual aabb getBoundingbox() const override final
		{
			return std::move(bbox);
		}
		
		std::vector<face> faces;
		std::vector<vertex> vertices;
		material* mtrl{ nullptr };
		aabb bbox;
	};

	class object {
		friend class ObjLoader;
		friend class objinstance;

	public:
		object() {}
		~object() {}

	private:
		std::vector<shape*> m_shapes;
		aabb m_aabb;
	};

	class objinstance : public bvhnode {
	public:
		objinstance() {}
		objinstance(object* obj);

		virtual ~objinstance() {}

	public:
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override final;

		virtual aabb getBoundingbox() const override final;

	private:
		object* m_obj{ nullptr };
	};
}
