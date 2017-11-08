#include "geometry/object.h"
#include "math/intersect.h"
#include "accelerator/accelerator.h"

#include <iterator>

//#define ENABLE_LINEAR_HITTEST

namespace AT_NAME
{
	std::atomic<int> face::s_id = 0;
	std::vector<face*> face::s_faces;

	face::face()
	{
		id = s_id.fetch_add(1);
		s_faces.push_back(this);
	}

	face::~face()
	{
		auto it = std::find(s_faces.begin(), s_faces.end(), this);
		if (it != s_faces.end()) {
			s_faces.erase(it);
		}
	}

	bool face::hit(
		const aten::ray& r,
		real t_min, real t_max,
		aten::Intersection& isect) const
	{
		const auto& v0 = aten::VertexManager::getVertex(param.idx[0]);
		const auto& v1 = aten::VertexManager::getVertex(param.idx[1]);
		const auto& v2 = aten::VertexManager::getVertex(param.idx[2]);

		bool isHit = hit(
			&param, 
			v0.pos, v1.pos, v2.pos,
			r, 
			t_min, t_max, 
			&isect);

		if (isHit) {
			// Temporary, notify triangle id to the parent object.
			isect.objid = id;

			isect.primid = id;

#if 0
			if (parent) {
				isect.mtrlid = ((material*)parent->param.mtrl.ptr)->id();
			}
#else
			isect.mtrlid = param.mtrlid;
#endif
		}

		return isHit;
	}

	bool face::hit(
		const aten::PrimitiveParamter* param,
		const aten::vec3& v0,
		const aten::vec3& v1,
		const aten::vec3& v2,
		const aten::ray& r,
		real t_min, real t_max,
		aten::Intersection* isect)
	{
		bool isHit = false;

		const auto res = intersectTriangle(r, v0, v1, v2);

		if (res.isIntersect) {
			if (res.t < isect->t) {
				isect->t = res.t;

				isect->a = res.a;
				isect->b = res.b;

				isect->area = param->area;

				isHit = true;
			}
		}

		return isHit;
	}

	void face::evalHitResult(
		const aten::ray& r, 
		aten::hitrecord& rec,
		const aten::Intersection& isect) const
	{
		const auto& v0 = aten::VertexManager::getVertex(param.idx[0]);
		const auto& v1 = aten::VertexManager::getVertex(param.idx[1]);
		const auto& v2 = aten::VertexManager::getVertex(param.idx[2]);

		evalHitResult(v0, v1, v2, &rec, &isect);

		if (param.needNormal > 0) {
			auto e01 = v1.pos - v0.pos;
			auto e02 = v2.pos - v0.pos;

			e01.w = e02.w = real(0);

			rec.normal = normalize(cross(e01, e02));
		}
	}

	void face::evalHitResult(
		const aten::vertex& v0,
		const aten::vertex& v1,
		const aten::vertex& v2,
		aten::hitrecord* rec,
		const aten::Intersection* isect)
	{
		// NOTE
		// http://d.hatena.ne.jp/Zellij/20131207/p1

		real a = isect->a;
		real b = isect->b;
		real c = 1 - a - b;

		// dSÀ•WŒn(barycentric coordinates).
		// v0Šî€.
		// p = (1 - a - b)*v0 + a*v1 + b*v2
		rec->p = c * v0.pos + a * v1.pos + b * v2.pos;
		rec->normal = c * v0.nml + a * v1.nml + b * v2.nml;
		auto uv = c * v0.uv + a * v1.uv + b * v2.uv;

		rec->u = uv.x;
		rec->v = uv.y;
	}

	void face::build(objshape* _parent)
	{
		const auto& v0 = aten::VertexManager::getVertex(param.idx[0]);
		const auto& v1 = aten::VertexManager::getVertex(param.idx[1]);
		const auto& v2 = aten::VertexManager::getVertex(param.idx[2]);

		aten::vec3 vmax = aten::vec3(
			std::max(v0.pos.x, std::max(v1.pos.x, v2.pos.x)),
			std::max(v0.pos.y, std::max(v1.pos.y, v2.pos.y)),
			std::max(v0.pos.z, std::max(v1.pos.z, v2.pos.z)));

		aten::vec3 vmin = aten::vec3(
			std::min(v0.pos.x, std::min(v1.pos.x, v2.pos.x)),
			std::min(v0.pos.y, std::min(v1.pos.y, v2.pos.y)),
			std::min(v0.pos.z, std::min(v1.pos.z, v2.pos.z)));

		setBoundingBox(aten::aabb(vmin, vmax));

		// ŽOŠpŒ`‚Ì–ÊÏ = ‚Q•Ó‚ÌŠOÏ‚Ì’·‚³ / 2;
		auto e0 = v1.pos - v0.pos;
		auto e1 = v2.pos - v0.pos;
		param.area = real(0.5) * cross(e0, e1).length();

		parent = _parent;
		param.mtrlid = ((material*)parent->param.mtrl.ptr)->id();
	}

	void face::getSamplePosNormalArea(
		aten::hitable::SamplePosNormalPdfResult* result,
		aten::sampler* sampler) const
	{
#if 0
		// 0 <= a + b <= 1
		real a = sampler->nextSample();
		real b = sampler->nextSample();

		real d = a + b;

		if (d > 1) {
			a /= d;
			b /= d;
		}
#else
		real r0 = sampler->nextSample();
		real r1 = sampler->nextSample();

		real a = aten::sqrt(r0) * (real(1) - r1);
		real b = aten::sqrt(r0) * r1;
#endif

		const auto& v0 = aten::VertexManager::getVertex(param.idx[0]);
		const auto& v1 = aten::VertexManager::getVertex(param.idx[1]);
		const auto& v2 = aten::VertexManager::getVertex(param.idx[2]);

		// dSÀ•WŒn(barycentric coordinates).
		// v0Šî€.
		// p = (1 - a - b)*v0 + a*v1 + b*v2
		aten::vec3 p = (1 - a - b) * v0.pos + a * v1.pos + b * v2.pos;
		
		aten::vec3 n = (1 - a - b) * v0.nml + a * v1.nml + b * v2.nml;
		n = normalize(n);

		// ŽOŠpŒ`‚Ì–ÊÏ = ‚Q•Ó‚ÌŠOÏ‚Ì’·‚³ / 2;
		auto e0 = v1.pos - v0.pos;
		auto e1 = v2.pos - v0.pos;
		auto area = real(0.5) * cross(e0, e1).length();

		result->pos = p;
		result->nml = n;
		result->area = area;

		result->a = a;
		result->b = b;

		result->primid = id;
	}

	int face::geomid() const
	{
		AT_ASSERT(parent);
		if (parent) {
			return parent->getGeomId();
		}
		return -1;
	}

	int face::findIdx(hitable* h)
	{
		int idx = -1;

		if (h) {
			auto found = std::find(s_faces.begin(), s_faces.end(), h);
			if (found != s_faces.end()) {
				idx = std::distance(s_faces.begin(), found);
				AT_ASSERT(h == s_faces[idx]);
			}
		}

		return idx;
	}


	void objshape::build()
	{
		aten::vec3 boxmin(AT_MATH_INF, AT_MATH_INF, AT_MATH_INF);
		aten::vec3 boxmax(-AT_MATH_INF, -AT_MATH_INF, -AT_MATH_INF);

		param.area = 0;

		for (const auto f : faces) {
			f->build(this);
			param.area += f->param.area;

			const auto& faabb = f->getBoundingbox();

			boxmin = aten::min(faabb.minPos(), boxmin);
			boxmax = aten::max(faabb.maxPos(), boxmax);
		}

		m_aabb.init(boxmin, boxmax);
	}

	void object::build()
	{
		if (m_triangles > 0) {
			// Builded already.
			return;
		}

		m_accel = aten::accelerator::createAccelerator();

		param.primid = shapes[0]->faces[0]->id;

		param.area = 0;
		m_triangles = 0;

		// Avoid sorting objshape list in bvh::build directly.
		std::vector<face*> tmp;

		for (const auto s : shapes) {
			s->build();

			param.area += s->param.area;
			m_triangles += (uint32_t)s->faces.size();

			tmp.insert(tmp.end(), s->faces.begin(), s->faces.end());
		}

		param.primnum = m_triangles;

		m_accel->build((hitable**)&tmp[0], (uint32_t)tmp.size());
		bbox = m_accel->getBoundingbox();
	}

	bool object::hit(
		const aten::ray& r,
		real t_min, real t_max,
		aten::Intersection& isect) const
	{
		bool isHit = m_accel->hit(r, t_min, t_max, isect);

		if (isHit) {
			auto f = face::faces()[isect.objid];

			// Ž©g‚ÌID‚ð•Ô‚·.
			isect.objid = id();
		}
		return isHit;
	}

	void object::evalHitResult(
		const aten::ray& r,
		const aten::mat4& mtxL2W,
		aten::hitrecord& rec,
		const aten::Intersection& isect) const
	{
		auto f = face::faces()[isect.primid];

		auto& vtxs = aten::VertexManager::getVertices();

		const auto& v0 = vtxs[f->param.idx[0]];
		const auto& v1 = vtxs[f->param.idx[1]];
		const auto& v2 = vtxs[f->param.idx[2]];

		//face::evalHitResult(v0, v1, v2, &rec, &isect);
		f->evalHitResult(r, rec, isect);

		real orignalLen = 0;
		{
			const auto& p0 = v0.pos;
			const auto& p1 = v1.pos;

			orignalLen = length(p1.v - p0.v);
		}

		real scaledLen = 0;
		{
			auto p0 = mtxL2W.apply(v0.pos);
			auto p1 = mtxL2W.apply(v1.pos);

			scaledLen = length(p1.v - p0.v);
		}

		real ratio = scaledLen / orignalLen;
		ratio = ratio * ratio;

		rec.area = param.area * ratio;

		rec.objid = isect.objid;
		rec.mtrlid = isect.mtrlid;
	}

	void object::getSamplePosNormalArea(
		aten::hitable::SamplePosNormalPdfResult* result,
		const aten::mat4& mtxL2W, 
		aten::sampler* sampler) const
	{
		auto r = sampler->nextSample();
		int shapeidx = (int)(r * (shapes.size() - 1));
		auto objshape = shapes[shapeidx];

		r = sampler->nextSample();
		int faceidx = (int)(r * (objshape->faces.size() - 1));
		auto f = objshape->faces[faceidx];

		const auto& v0 = aten::VertexManager::getVertex(f->param.idx[0]);
		const auto& v1 = aten::VertexManager::getVertex(f->param.idx[1]);

		real orignalLen = 0;
		{
			const auto& p0 = v0.pos;
			const auto& p1 = v1.pos;

			orignalLen = (p1 - p0).length();
		}

		real scaledLen = 0;
		{
			auto p0 = mtxL2W.apply(v0.pos);
			auto p1 = mtxL2W.apply(v1.pos);

			scaledLen = length(p1.v - p0.v);
		}

		real ratio = scaledLen / orignalLen;
		ratio = ratio * ratio;

		auto area = param.area * ratio;

		f->getSamplePosNormalArea(result, sampler);

		result->area = area;
	}

	void object::getPrimitives(aten::PrimitiveParamter* primparams) const
	{
		int cnt = 0;

		for (auto s : shapes) {
			const auto& shapeParam = s->param;
			
			auto mtrlid = material::findMaterialIdx((material*)shapeParam.mtrl.ptr);

			for (auto f : s->faces) {
				auto faceParam = f->param;
				faceParam.mtrlid = mtrlid;
				primparams[cnt++] = faceParam;
			}
		}
	}
}
