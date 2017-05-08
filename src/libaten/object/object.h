#pragma once

#include <atomic>

#include "types.h"
#include "accelerator/bvh.h"
#include "material/material.h"
#include "math/mat4.h"
#include "shape/shape.h"
#include "shape/tranformable.h"
#include "object/vertex.h"

namespace aten
{
	class shape;

	class face : public bvhnode {
		static std::atomic<int> s_id;
		static std::vector<face*> s_faces;

	public:
		face();
		virtual ~face();

	public:
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override;

		static bool hit(
			const PrimitiveParamter& param,
			const vertex& v0,
			const vertex& v1,
			const vertex& v2,
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec);

		virtual vec3 getRandomPosOn(sampler* sampler) const override;

		virtual SamplingPosNormalPdf getSamplePosNormalPdf(sampler* sampler) const override;

		void build();

		static const std::vector<face*>& faces()
		{
			return s_faces;
		}
	
		PrimitiveParamter param;
		shape* parent{ nullptr };
		int id{ -1 };
	};

	class shape : public bvhnode {
	public:
		shape() : param(ShapeType::Polygon) {}
		virtual ~shape() {}

		void build();

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override final;

		virtual vec3 getRandomPosOn(sampler* sampler) const override final
		{
			auto r = sampler->nextSample();
			int idx = (int)(r * (faces.size() - 1));
			auto face = faces[idx];
			return face->getRandomPosOn(sampler);
		}

		virtual SamplingPosNormalPdf getSamplePosNormalPdf(sampler* sampler) const override final
		{
			auto r = sampler->nextSample();
			int idx = (int)(r * (faces.size() - 1));
			auto face = faces[idx];
			return face->getSamplePosNormalPdf(sampler);
		}

		ShapeParameter param;		
		std::vector<face*> faces;

	private:
		bvhnode m_node;
	};

	template<typename T> class instance;

	class object : public transformable {
		friend class instance<object>;

	public:
		object() : param(ShapeType::Polygon) {}
		virtual ~object() {}

	public:
		virtual bool hit(
			const ray& r,
			const mat4& mtxL2W,
			real t_min, real t_max,
			hitrecord& rec) const override final;

		virtual void getPrimitives(std::vector<PrimitiveParamter>& primparams) const override final;

		virtual const ShapeParameter& getParam() const override final
		{
			return param;
		}

	private:
		void build();

		vec3 getRandomPosOn(sampler* sampler) const
		{
			auto r = sampler->nextSample();
			int idx = (int)(r * (shapes.size() - 1));
			auto shape = shapes[idx];
			return shape->getRandomPosOn(sampler);
		}

		virtual hitable::SamplingPosNormalPdf getSamplePosNormalPdf(const mat4& mtxL2W, sampler* sampler) const override final;

		virtual int collectInternalNodes(std::vector<BVHNode>& nodes, int order, bvhnode* parent) override final;

	public:
		std::vector<shape*> shapes;
		ShapeParameter param;
		aabb bbox;

	private:
		bvhnode m_node;
		uint32_t m_triangles{ 0 };
	};
}
