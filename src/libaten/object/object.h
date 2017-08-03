#pragma once

#include <atomic>

#include "types.h"
#include "accelerator/bvh.h"
#include "material/material.h"
#include "math/mat4.h"
#include "shape/shape.h"
#include "shape/tranformable.h"
#include "shape/mesh.h"
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
			aten::Intersection& isect) const override;

		static bool hit(
			const aten::PrimitiveParamter* param,
			const aten::vec3& v0,
			const aten::vec3& v1,
			const aten::vec3& v2,
			const aten::ray& r,
			real t_min, real t_max,
			aten::Intersection* isect);

		virtual void evalHitResult(
			const aten::ray& r, 
			aten::hitrecord& rec,
			const aten::Intersection& isect) const;

		static void evalHitResult(
			const aten::vertex& v0,
			const aten::vertex& v1,
			const aten::vertex& v2,
			aten::hitrecord* rec,
			const aten::Intersection* isect);

		virtual void getSamplePosNormalArea(
			aten::hitable::SamplePosNormalPdfResult* result,
			aten::sampler* sampler) const override;

		void build();

		static const std::vector<face*>& faces()
		{
			return s_faces;
		}
	
		aten::PrimitiveParamter param;
		shape* parent{ nullptr };
		int id{ -1 };

	private:
		virtual bool setBVHNodeParam(
			aten::BVHNode& param,
			const bvhnode* parent,
			const int idx,
			std::vector<std::vector<aten::BVHNode>>& nodes,
			const aten::transformable* instanceParent,
			const aten::mat4& mtxL2W) override final
		{
			bvhnode::setBVHNodeParam(param, parent, idx, nodes, instanceParent, mtxL2W);
			param.primid = (float)id;
			param.exid = -1;
			return true;
		}
	};

	class shape : public aten::mesh<aten::bvhnode> {
		friend class object;

	public:
		shape() : param(aten::ShapeType::Polygon) {}
		virtual ~shape() {}

		void build();

		virtual bool hit(
			const aten::ray& r,
			real t_min, real t_max,
			aten::Intersection& isect) const override final;

		virtual void getSamplePosNormalArea(
			aten::hitable::SamplePosNormalPdfResult* result,
			aten::sampler* sampler) const override final
		{
			auto r = sampler->nextSample();
			int idx = (int)(r * (faces.size() - 1));
			auto face = faces[idx];
			return face->getSamplePosNormalArea(result, sampler);
		}

		void setMaterial(material* mtrl)
		{
			param.mtrl.ptr = mtrl;
			m_mtrl = mtrl;
		}

		const material* getMaterial() const
		{
			return m_mtrl;
		}

		aten::ShapeParameter param;
		std::vector<face*> faces;

	private:
		bvhnode m_node;
		material* m_mtrl{ nullptr };
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
			real t_min, real t_max,
			aten::Intersection& isect) const override final;

		virtual void evalHitResult(
			const aten::ray& r,
			const aten::mat4& mtxL2W,
			aten::hitrecord& rec,
			const aten::Intersection& isect) const override final;

		virtual void getPrimitives(aten::PrimitiveParamter* primparams) const override final;

		virtual const aten::ShapeParameter& getParam() const override final
		{
			return param;
		}

	private:
		void build();

		virtual void getSamplePosNormalArea(
			aten::hitable::SamplePosNormalPdfResult* result,
			const aten::mat4& mtxL2W, 
			aten::sampler* sampler) const override final;

		virtual bool setBVHNodeParam(
			aten::BVHNode& param,
			const bvhnode* parent,
			const int idx,
			std::vector<std::vector<aten::BVHNode>>& nodes,
			const transformable* instanceParent,
			const aten::mat4& mtxL2W) override final;

		virtual void registerToList(
			const int idx,
			std::vector<std::vector<bvhnode*>>& nodeList) override final;

		virtual bvhnode* getNode() override final;

	public:
		std::vector<shape*> shapes;
		aten::ShapeParameter param;
		aten::aabb bbox;

	private:
		bvhnode m_node;
		uint32_t m_triangles{ 0 };
	};
}
