#pragma once

#include "types.h"
#include "scene/bvh.h"
#include "material/material.h"
#include "math/mat4.h"

namespace aten
{
	struct vertex {
		vec3 pos;
		vec3 nml;
		vec3 uv;
	};

	class shape;

	class face : public bvhnode {
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

		shape* parent{ nullptr };
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

	private:
		bvhnode m_node;
	};

	class object {
		friend class objinstance;

	public:
		object() {}
		~object() {}

	public:
		bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec)
		{
			bool isHit = m_node.hit(r, t_min, t_max, rec);
			return isHit;
		}

	private:
		void build()
		{
			m_node.build((bvhnode**)&shapes[0], shapes.size());
		}

	public:
		std::vector<shape*> shapes;
		aabb bbox;

	private:
		bvhnode m_node;
	};

	class objinstance : public bvhnode {
	public:
		objinstance() {}
		objinstance(object* obj);
		objinstance(object* obj, const mat4& mtxL2W)
			: objinstance(obj)
		{
			m_mtxL2W = mtxL2W;

			m_mtxW2L = m_mtxL2W;
			m_mtxW2L.invert();

			m_aabb = transformBoundingBox();
		}

		virtual ~objinstance() {}

	public:
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override final;

		virtual aabb getBoundingbox() const override final;

	private:
		aabb transformBoundingBox();

	private:
		object* m_obj{ nullptr };
		mat4 m_mtxL2W;
		mat4 m_mtxW2L;	// inverted.
	};
}
