#include "object/object.h"
#include "math/intersect.h"

#include <iterator>

//#define ENABLE_LINEAR_HITTEST
//#define ENABLE_DIRECT_FACE_BVH

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
		aten::hitrecord& rec) const
	{
		const auto& v0 = aten::VertexManager::getVertex(param.idx[0]);
		const auto& v1 = aten::VertexManager::getVertex(param.idx[1]);
		const auto& v2 = aten::VertexManager::getVertex(param.idx[2]);

		bool isHit = hit(
			&param, 
			v0, v1, v2, 
			r, 
			t_min, t_max, 
			&rec);

		if (isHit) {
			//rec.obj = parent;
			rec.obj = (hitable*)this;

			rec.param.idx[0] = param.idx[0];
			rec.param.idx[1] = param.idx[1];
			rec.param.idx[2] = param.idx[2];

			if (parent) {
				rec.mtrl = (material*)parent->param.mtrl.ptr;
			}
		}

		return isHit;
	}

	bool face::hit(
		const aten::PrimitiveParamter* param,
		const aten::vertex& v0,
		const aten::vertex& v1,
		const aten::vertex& v2,
		const aten::ray& r,
		real t_min, real t_max,
		aten::hitrecord* rec)
	{
		bool isHit = false;

		const auto res = intersectTriangle(r, v0.pos, v1.pos, v2.pos);

		if (res.isIntersect) {
			if (res.t < rec->t) {
				rec->t = res.t;

				rec->param.a = res.a;
				rec->param.b = res.b;

				rec->area = param->area;

				isHit = true;
			}
		}

		return isHit;
	}

	AT_DEVICE_API void face::evalHitResult(
		const aten::vertex& v0,
		const aten::vertex& v1,
		const aten::vertex& v2,
		aten::hitrecord* rec)
	{
		// NOTE
		// http://d.hatena.ne.jp/Zellij/20131207/p1

		real a = rec->param.a;
		real b = rec->param.b;
		real c = 1 - a - b;

		// 重心座標系(barycentric coordinates).
		// v0基準.
		// p = (1 - a - b)*v0 + a*v1 + b*v2
		rec->p = c * v0.pos + a * v1.pos + b * v2.pos;
		rec->normal = c * v0.nml + a * v1.nml + b * v2.nml;
		auto uv = c * v0.uv + a * v1.uv + b * v2.uv;

		rec->u = uv.x;
		rec->v = uv.y;

		// tangent coordinate.
		rec->du = normalize(getOrthoVector(rec->normal));
		rec->dv = normalize(cross(rec->normal, rec->du));
	}

	void face::build()
	{
		const auto& v0 = aten::VertexManager::getVertex(param.idx[0]);
		const auto& v1 = aten::VertexManager::getVertex(param.idx[1]);
		const auto& v2 = aten::VertexManager::getVertex(param.idx[2]);

		aten::vec3 vmax(
			std::max(v0.pos.x, std::max(v1.pos.x, v2.pos.x)),
			std::max(v0.pos.y, std::max(v1.pos.y, v2.pos.y)),
			std::max(v0.pos.z, std::max(v1.pos.z, v2.pos.z)));

		aten::vec3 vmin(
			std::min(v0.pos.x, std::min(v1.pos.x, v2.pos.x)),
			std::min(v0.pos.y, std::min(v1.pos.y, v2.pos.y)),
			std::min(v0.pos.z, std::min(v1.pos.z, v2.pos.z)));

		m_aabb.init(vmin, vmax);

		// 三角形の面積 = ２辺の外積の長さ / 2;
		auto e0 = v1.pos - v0.pos;
		auto e1 = v2.pos - v0.pos;
		param.area = real(0.5) * cross(e0, e1).length();
	}

	aten::vec3 face::getRandomPosOn(aten::sampler* sampler) const
	{
		// 0 <= a + b <= 1
		real a = sampler->nextSample();
		real b = sampler->nextSample();

		real d = a + b;

		if (d > 1) {
			a /= d;
			b /= d;
		}

		const auto& v0 = aten::VertexManager::getVertex(param.idx[0]);
		const auto& v1 = aten::VertexManager::getVertex(param.idx[1]);
		const auto& v2 = aten::VertexManager::getVertex(param.idx[2]);

		// 重心座標系(barycentric coordinates).
		// v0基準.
		// p = (1 - a - b)*v0 + a*v1 + b*v2
		aten::vec3 p = (1 - a - b) * v0.pos + a * v1.pos + b * v2.pos;

		return std::move(p);
	}

	aten::hitable::SamplingPosNormalPdf face::getSamplePosNormalPdf(aten::sampler* sampler) const
	{
		// 0 <= a + b <= 1
		real a = sampler->nextSample();
		real b = sampler->nextSample();

		real d = a + b;

		if (d > 1) {
			a /= d;
			b /= d;
		}

		const auto& v0 = aten::VertexManager::getVertex(param.idx[0]);
		const auto& v1 = aten::VertexManager::getVertex(param.idx[1]);
		const auto& v2 = aten::VertexManager::getVertex(param.idx[2]);

		// 重心座標系(barycentric coordinates).
		// v0基準.
		// p = (1 - a - b)*v0 + a*v1 + b*v2
		aten::vec3 p = (1 - a - b) * v0.pos + a * v1.pos + b * v2.pos;
		
		aten::vec3 n = (1 - a - b) * v0.nml + a * v1.nml + b * v2.nml;
		n.normalize();

		// 三角形の面積 = ２辺の外積の長さ / 2;
		auto e0 = v1.pos - v0.pos;
		auto e1 = v2.pos - v0.pos;
		auto area = real(0.5) * cross(e0, e1).length();

		return std::move(hitable::SamplingPosNormalPdf(p + n * AT_MATH_EPSILON, n, area));
	}

	void shape::build()
	{
#ifndef ENABLE_DIRECT_FACE_BVH
		// Avoid sorting face list in bvh::build directly.
		std::vector<face*> tmp;
		std::copy(faces.begin(), faces.end(), std::back_inserter(tmp));

		m_node.build(
			(bvhnode**)&tmp[0],
			(uint32_t)tmp.size());

		m_aabb = m_node.getBoundingbox();
#endif

		param.area = 0;
		for (const auto f : faces) {
			param.area += f->param.area;

#ifdef ENABLE_DIRECT_FACE_BVH
			m_aabb = aten::aabb::merge(m_aabb, f->getBoundingbox());
#endif
		}
	}

	bool shape::hit(
		const aten::ray& r,
		real t_min, real t_max,
		aten::hitrecord& rec) const
	{
#ifdef ENABLE_LINEAR_HITTEST
		bool isHit = false;

		hitrecord tmp;

		for (auto f : faces) {
			hitrecord tmptmp;
			if (f->hit(r, t_min, t_max, tmptmp)) {
				if (tmptmp.t < tmp.t) {
					tmp = tmptmp;
					isHit = true;
				}
			}
		}

		if (isHit) {
			rec = tmp;
		}
#else
		auto isHit = m_node.hit(r, t_min, t_max, rec);
#endif

		if (isHit) {
			rec.mtrl = (material*)param.mtrl.ptr;
		}

		return isHit;
	}

	void object::build()
	{
		if (m_triangles > 0) {
			// Builded already.
			return;
		}

		param.primid = shapes[0]->faces[0]->id;

		param.area = 0;
		m_triangles = 0;

		for (const auto s : shapes) {
			s->build();

			param.area += s->param.area;
			m_triangles += (uint32_t)s->faces.size();
		}

#ifdef ENABLE_DIRECT_FACE_BVH
		std::vector<face*> faces;
		for (auto s : shapes) {
			bbox = aten::aabb::merge(bbox, s->getBoundingbox());
			for (auto f : s->faces) {
				faces.push_back(f);
			}
		}
		m_node.build((bvhnode**)&faces[0], (uint32_t)faces.size());
#else
		// Avoid sorting shape list in bvh::build directly.
		std::vector<shape*> tmp;
		std::copy(shapes.begin(), shapes.end(), std::back_inserter(tmp));

		m_node.build((bvhnode**)&tmp[0], (uint32_t)tmp.size());
		bbox = m_node.getBoundingbox();
#endif
	}

	bool object::hit(
		const aten::ray& r,
		const aten::mat4& mtxL2W,
		real t_min, real t_max,
		aten::hitrecord& rec) const
	{
#ifdef ENABLE_LINEAR_HITTEST
		bool isHit = false;

		hitrecord tmp;

		for (auto s : shapes) {
			hitrecord tmptmp;

			if (s->hit(r, t_min, t_max, tmptmp)) {
				if (tmptmp.t < tmp.t) {
					tmp = tmptmp;
					isHit = true;
				}
			}
		}
#else
		aten::hitrecord tmp;
		bool isHit = m_node.hit(r, t_min, t_max, tmp);
#endif

		if (isHit) {
			rec = tmp;

			face* f = (face*)rec.obj;

#if 0
			real originalArea = 0;
			{
				const auto& v0 = f->vtx[0]->pos;
				const auto& v1 = f->vtx[1]->pos;
				const auto& v2 = f->vtx[2]->pos;

				// 三角形の面積 = ２辺の外積の長さ / 2;
				auto e0 = v1 - v0;
				auto e1 = v2 - v0;
				originalArea = 0.5 * cross(e0, e1).length();
			}

			real scaledArea = 0;
			{
				auto v0 = mtxL2W.apply(f->vtx[0]->pos);
				auto v1 = mtxL2W.apply(f->vtx[1]->pos);
				auto v2 = mtxL2W.apply(f->vtx[2]->pos);

				// 三角形の面積 = ２辺の外積の長さ / 2;
				auto e0 = v1 - v0;
				auto e1 = v2 - v0;
				scaledArea = 0.5 * cross(e0, e1).length();
			}

			real ratio = scaledArea / originalArea;
#else
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

				scaledLen = (p1 - p0).length();
			}

			real ratio = scaledLen / orignalLen;
			ratio = ratio * ratio;
#endif

			rec.area = param.area * ratio;

			// 最終的には、やっぱりshapeを渡す.
			rec.obj = f->parent;
		}
		return isHit;
	}

	void object::evalHitResult(
		const aten::ray& r,
		const aten::mat4& mtxL2W,
		aten::hitrecord& rec) const
	{
		const auto& v0 = aten::VertexManager::getVertex(rec.param.idx[0]);
		const auto& v1 = aten::VertexManager::getVertex(rec.param.idx[1]);
		const auto& v2 = aten::VertexManager::getVertex(rec.param.idx[2]);

		face::evalHitResult(v0, v1, v2, &rec);
	}

	aten::hitable::SamplingPosNormalPdf object::getSamplePosNormalPdf(const aten::mat4& mtxL2W, aten::sampler* sampler) const
	{
		auto r = sampler->nextSample();
		int shapeidx = (int)(r * (shapes.size() - 1));
		auto shape = shapes[shapeidx];

		r = sampler->nextSample();
		int faceidx = (int)(r * (shape->faces.size() - 1));
		auto f = shape->faces[faceidx];

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

			scaledLen = (p1 - p0).length();
		}

		real ratio = scaledLen / orignalLen;
		ratio = ratio * ratio;

		auto area = param.area * ratio;

		auto tmp = f->getSamplePosNormalPdf(sampler);

		hitable::SamplingPosNormalPdf res(
			std::get<0>(tmp),
			std::get<1>(tmp),
			real(1) / area);

		return std::move(res);
	}

	void object::getPrimitives(std::vector<aten::PrimitiveParamter>& primparams) const
	{
		for (auto s : shapes) {
			const auto& shapeParam = s->param;
			
			auto mtrlid = material::findMaterialIdx((material*)shapeParam.mtrl.ptr);

			for (auto f : s->faces) {
				auto faceParam = f->param;
				faceParam.mtrlid = mtrlid;
				primparams.push_back(faceParam);
			}
		}
	}

	int object::collectInternalNodes(std::vector<aten::BVHNode>& nodes, int order, bvhnode* parent)
	{
		int newOrder = aten::bvh::setTraverseOrder(&m_node, order);
		aten::bvh::collectNodes(&m_node, nodes, parent);

#ifndef ENABLE_DIRECT_FACE_BVH
		for (auto s : shapes) {
			const auto idx = s->m_traverseOrder;
			auto& node = nodes[idx];
			node.nestid = nodes.size();

			newOrder = aten::bvh::setTraverseOrder(&s->m_node, newOrder);
			aten::bvh::collectNodes(&s->m_node, nodes, parent);
		}
#endif

		return newOrder;
	}
}
