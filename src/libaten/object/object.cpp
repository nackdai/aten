#include "object/object.h"
#include "math/intersect.h"

namespace aten
{
	bool face::hit(
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec) const
	{
		bool isHit = false;

#if 0
		const auto v0 = vtx[idx[0]];
		const auto v1 = vtx[idx[1]];
		const auto v2 = vtx[idx[2]];
#else
		const auto v0 = vtx[0];
		const auto v1 = vtx[1];
		const auto v2 = vtx[2];
#endif

		const auto res = intersertTriangle(r, v0->pos, v1->pos, v2->pos);

		if (res.isIntersect) {
			if (res.t < rec.t) {
				rec.t = res.t;

				auto p = r.org + rec.t * r.dir;

				// NOTE
				// http://d.hatena.ne.jp/Zellij/20131207/p1

				// dSÀ•WŒn(barycentric coordinates).
				// v0Šî€.
				// p = (1 - a - b)*v0 + a*v1 + b*v2
				rec.p = (1 - res.a - res.b) * v0->pos + res.a * v1->pos + res.b * v2->pos;
				rec.normal = (1 - res.a - res.b) * v0->nml + res.a * v1->nml + res.b * v2->nml;
				auto uv = (1 - res.a - res.b) * v0->uv + res.a * v1->uv + res.b * v2->uv;

				rec.u = uv.x;
				rec.v = uv.y;

				// tangent coordinate.
				rec.du = normalize(getOrthoVector(rec.normal));
				rec.dv = normalize(cross(rec.normal, rec.du));

				// ŽOŠpŒ`‚Ì–ÊÏ = ‚Q•Ó‚ÌŠOÏ‚Ì’·‚³ / 2;
				auto e0 = v1->pos - v0->pos;
				auto e1 = v2->pos - v0->pos;
				rec.area = 0.5 * cross(e0, e1).length();

				rec.obj = parent;

				if (parent) {
					rec.mtrl = parent->mtrl;
				}

				isHit = true;
			}
		}

		return isHit;
	}

	void face::build(vertex* v0, vertex* v1, vertex* v2)
	{
		vec3 vmax(
			std::max(v0->pos.x, std::max(v1->pos.x, v2->pos.x)),
			std::max(v0->pos.y, std::max(v1->pos.y, v2->pos.y)),
			std::max(v0->pos.z, std::max(v1->pos.z, v2->pos.z)));

		vec3 vmin(
			std::min(v0->pos.x, std::min(v1->pos.x, v2->pos.x)),
			std::min(v0->pos.y, std::min(v1->pos.y, v2->pos.y)),
			std::min(v0->pos.z, std::min(v1->pos.z, v2->pos.z)));

		bbox.init(vmin, vmax);

		vtx[0] = v0;
		vtx[1] = v1;
		vtx[2] = v2;
	}

	void shape::build()
	{
		m_node.build(
			(bvhnode**)&faces[0],
			faces.size());

		m_aabb = m_node.getBoundingbox();
	}

	bool shape::hit(
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec) const
	{
		auto isHit = m_node.hit(r, t_min, t_max, rec);

		if (isHit) {
			rec.obj = (hitable*)this;
			rec.mtrl = mtrl;
		}

		return isHit;
	}

	objinstance::objinstance(object* obj)
	{
		m_obj = obj;
		m_obj->build();
		m_aabb = m_obj->m_aabb;
	}

	bool objinstance::hit(
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec) const
	{
		vec3 org = r.org;
		vec3 dir = r.dir;

		org = m_mtxW2L.apply(org);
		dir = m_mtxW2L.applyXYZ(dir);

		ray transformdRay(org, dir);

		auto isHit = m_obj->hit(transformdRay, t_min, t_max, rec);

		if (isHit) {
			// Transform local to world.
			rec.p = m_mtxL2W.apply(rec.p);
			rec.normal = normalize(m_mtxL2W.applyXYZ(rec.normal));
		}

		return isHit;
	}

	aabb objinstance::getBoundingbox() const
	{
		return std::move(m_aabb);
	}

	aabb objinstance::transformBoundingBox()
	{
		auto box = m_obj->m_aabb;

		vec3 center = box.getCenter();

		vec3 vMin = box.minPos() - center;
		vec3 vMax = box.maxPos() - center;

		vec3 pts[8] = {
			vec3(vMin.x, vMin.y, vMin.z),
			vec3(vMax.x, vMin.y, vMin.z),
			vec3(vMin.x, vMax.y, vMin.z),
			vec3(vMax.x, vMax.y, vMin.z),
			vec3(vMin.x, vMin.y, vMax.z),
			vec3(vMax.x, vMin.y, vMax.z),
			vec3(vMin.x, vMax.y, vMax.z),
			vec3(vMax.x, vMax.y, vMax.z),
		};

		vec3 newMin(AT_MATH_INF);
		vec3 newMax(-AT_MATH_INF);

		for (int i = 0; i < 8; i++) {
			vec3 v = m_mtxL2W.apply(pts[i]);

			newMin.set(
				std::min(newMin.x, v.x),
				std::min(newMin.y, v.y),
				std::min(newMin.z, v.z));
			newMax.set(
				std::max(newMax.x, v.x),
				std::max(newMax.y, v.y),
				std::max(newMax.z, v.z));
		}

		// Add only translate factor.
		vec3 newCenter = center + m_mtxL2W.apply(vec3());

		newMin += newCenter;
		newMax += newCenter;

		aabb ret(newMin, newMax);

		return std::move(ret);
	}
}
