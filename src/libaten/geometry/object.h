#pragma once

#include <atomic>

#include "types.h"
#include "material/material.h"
#include "math/mat4.h"
#include "geometry/geomparam.h"
#include "geometry/tranformable.h"
#include "geometry/geombase.h"
#include "geometry/vertex.h"

namespace AT_NAME
{
	class objshape;

	class face : public aten::hitable {
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

		virtual int geomid() const override;

		void build(objshape* _parent);

		static const std::vector<face*>& faces()
		{
			return s_faces;
		}

		static int findIdx(hitable* h);
	
		aten::PrimitiveParamter param;
		objshape* parent{ nullptr };
		int id{ -1 };
	};

	class objshape : public aten::geombase {
		friend class object;

	public:
		objshape() : param(aten::GeometryType::Polygon) {}
		virtual ~objshape() {}

		void build();

		void setMaterial(material* mtrl)
		{
			param.mtrl.ptr = mtrl;
			m_mtrl = mtrl;
		}

		const material* getMaterial() const
		{
			return m_mtrl;
		}

		aten::GeomParameter param;
		std::vector<face*> faces;
		aten::aabb m_aabb;

	private:
		material* m_mtrl{ nullptr };
	};

	template<typename T> class instance;

	class object : public aten::transformable {
		friend class instance<object>;

	public:
		object() : param(aten::GeometryType::Polygon) {}
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

		virtual const aten::GeomParameter& getParam() const override final
		{
			return param;
		}

		virtual aten::accelerator* getInternalAccelerator() override final
		{
			return m_accel;
		}

	private:
		void build();

		virtual void getSamplePosNormalArea(
			aten::hitable::SamplePosNormalPdfResult* result,
			const aten::mat4& mtxL2W, 
			aten::sampler* sampler) const override final;

	public:
		std::vector<objshape*> shapes;
		aten::GeomParameter param;
		aten::aabb bbox;

	private:
		aten::accelerator* m_accel{ nullptr };
		uint32_t m_triangles{ 0 };
	};
}
