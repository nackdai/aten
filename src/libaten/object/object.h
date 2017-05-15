#pragma once

#include <atomic>

#include "types.h"
#include "accelerator/bvh.h"
#include "material/material.h"
#include "math/mat4.h"
#include "shape/shape.h"
#include "shape/tranformable.h"
#include "object/vertex.h"

namespace AT_NAME
{
	class shape;

	class face : public aten::bvhnode {
		static std::atomic<int> s_id;
		static std::vector<face*> s_faces;

	public:
		face();
		virtual ~face();

	public:
		virtual bool hit(
			const aten::ray& r,
			real t_min, real t_max,
			aten::hitrecord& rec) const override;

		static AT_DEVICE_API bool hit(
			const aten::PrimitiveParamter* param,
			const aten::vertex& v0,
			const aten::vertex& v1,
			const aten::vertex& v2,
			const aten::ray& r,
			real t_min, real t_max,
			aten::hitrecord* rec);

		virtual void evalHitResult(const aten::ray& r, aten::hitrecord& rec) const;

		static AT_DEVICE_API void evalHitResult(
			const aten::vertex& v0,
			const aten::vertex& v1,
			const aten::vertex& v2,
			aten::hitrecord* rec);

		virtual aten::vec3 getRandomPosOn(aten::sampler* sampler) const override;

		virtual SamplingPosNormalPdf getSamplePosNormalPdf(aten::sampler* sampler) const override;

		void build();

		static const std::vector<face*>& faces()
		{
			return s_faces;
		}
	
		aten::PrimitiveParamter param;
		shape* parent{ nullptr };
		int id{ -1 };

	private:
		virtual void setBVHNodeParamInCollectNodes(aten::BVHNode& param) override final
		{
			param.primid = id;
		}
	};

	class shape : public aten::bvhnode {
		friend class object;

	public:
		shape() : param(aten::ShapeType::Polygon) {}
		virtual ~shape() {}

		void build();

		virtual bool hit(
			const aten::ray& r,
			real t_min, real t_max,
			aten::hitrecord& rec) const override final;

		virtual aten::vec3 getRandomPosOn(aten::sampler* sampler) const override final
		{
			auto r = sampler->nextSample();
			int idx = (int)(r * (faces.size() - 1));
			auto face = faces[idx];
			return face->getRandomPosOn(sampler);
		}

		virtual SamplingPosNormalPdf getSamplePosNormalPdf(aten::sampler* sampler) const override final
		{
			auto r = sampler->nextSample();
			int idx = (int)(r * (faces.size() - 1));
			auto face = faces[idx];
			return face->getSamplePosNormalPdf(sampler);
		}

		aten::ShapeParameter param;
		std::vector<face*> faces;

	private:
		bvhnode m_node;
	};

	template<typename T> class instance;

	class object : public aten::transformable {
		friend class instance<object>;

	public:
		object() : param(aten::ShapeType::Polygon) {}
		virtual ~object() {}

	public:
		virtual bool hit(
			const aten::ray& r,
			const aten::mat4& mtxL2W,
			real t_min, real t_max,
			aten::hitrecord& rec) const override final;

		virtual void evalHitResult(
			const aten::ray& r,
			const aten::mat4& mtxL2W,
			aten::hitrecord& rec) const override final;

		virtual void getPrimitives(std::vector<aten::PrimitiveParamter>& primparams) const override final;

		virtual const aten::ShapeParameter& getParam() const override final
		{
			return param;
		}

	private:
		void build();

		aten::vec3 getRandomPosOn(aten::sampler* sampler) const
		{
			auto r = sampler->nextSample();
			int idx = (int)(r * (shapes.size() - 1));
			auto shape = shapes[idx];
			return shape->getRandomPosOn(sampler);
		}

		virtual hitable::SamplingPosNormalPdf getSamplePosNormalPdf(const aten::mat4& mtxL2W, aten::sampler* sampler) const override final;

		virtual int collectInternalNodes(std::vector<aten::BVHNode>& nodes, int order, bvhnode* parent) override final;

	public:
		std::vector<shape*> shapes;
		aten::ShapeParameter param;
		aten::aabb bbox;

	private:
		bvhnode m_node;
		uint32_t m_triangles{ 0 };
	};
}
