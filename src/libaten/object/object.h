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

	class face : public hitable {
	public:
		face() {}
		virtual ~face() {}

	public:
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override;

		virtual aabb getBoundingbox() const override
		{
			return std::move(bbox);
		}

		void build(vertex* v0, vertex* v1, vertex* v2);
	
		uint32_t idx[3];
		vertex* vtx[3];
		aabb bbox;
	};

	class shape : public bvhnode {
	public:
		shape() {}
		virtual ~shape() {}

		void build();

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override final;
		
		std::vector<face*> faces;
		std::vector<vertex> vertices;
		material* mtrl{ nullptr };
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
