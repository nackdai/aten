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

		virtual vec3 getRandomPosOn(sampler* sampler) const override;

		void build(vertex* v0, vertex* v1, vertex* v2);
	
		uint32_t idx[3];
		vertex* vtx[3];
		aabb bbox;
		real area;

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

		virtual vec3 getRandomPosOn(sampler* sampler) const override final
		{
			auto r = sampler->nextSample();
			int idx = r * faces.size();
			auto face = faces[idx];
			return face->getRandomPosOn(sampler);
		}
		
		std::vector<face*> faces;
		std::vector<vertex> vertices;
		material* mtrl{ nullptr };
		real area;

	private:
		bvhnode m_node;
	};

	template<typename T> class instance;

	class object {
		friend class instance<object>;

	public:
		object() {}
		~object() {}

	public:
		bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec);

	private:
		void build()
		{
			m_node.build((bvhnode**)&shapes[0], (uint32_t)shapes.size());

			m_area = 0;
			for (const auto s : shapes) {
				m_area += s->area;
			}
		}

		vec3 getRandomPosOn(sampler* sampler) const
		{
			auto r = sampler->nextSample();
			int idx = r * shapes.size();
			auto shape = shapes[idx];
			return shape->getRandomPosOn(sampler);
		}

	public:
		std::vector<shape*> shapes;
		aabb bbox;

	private:
		bvhnode m_node;
		real m_area;
	};
}
