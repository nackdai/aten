#include "shape/cube.h"

namespace aten
{
	cube::cube(const vec3& c, real w, real h, real d, material* m)
		: m_center(c), m_size(w, h, d), m_mtrl(m)
	{
		m_bbox.init(
			m_center - m_size * 0.5,
			m_center + m_size * 0.5);
	}

	cube::Face cube::findFace(const vec3& d)
	{
		auto x = aten::abs(d.x);
		auto y = aten::abs(d.y);
		auto z = aten::abs(d.z);

		if (x > y && x > z) {
			// X軸に平行な面.
			if (d.x > 0) {
				return Face::POS_X;
			}
			else {
				return Face::NEG_X;
			}
		}
		else if (y > x && y > z) {
			// Y軸に平行な面.
			if (d.y > 0) {
				return Face::POS_Y;
			}
			else {
				return Face::NEG_Y;
			}
		}
		else {
			// Z軸に平行な面.
			if (d.z > 0) {
				return Face::POS_Z;
			}
			else {
				return Face::NEG_Z;
			}
		}
	}

	bool cube::hit(
		const ray& r, 
		real t_min, real t_max,
		hitrecord& rec) const
	{
		real t = 0;
		bool isHit = m_bbox.hit(r, t_min, t_max, &t);

		if (isHit) {
			rec.p = r.org + t * r.dir;

			rec.t = t;

			// どの面にヒットしたか探す.
			{
				auto dir = normalize(rec.p - m_center);
				auto face = findFace(dir);

				switch (face) {
				case POS_X:
					rec.normal = vec3(1, 0, 0);
					break;
				case NEG_X:
					rec.normal = vec3(-1, 0, 0);
					break;
				case POS_Y:
					rec.normal = vec3(0, 1, 0);
					break;
				case NEG_Y:
					rec.normal = vec3(0, -1, 0);
					break;
				case POS_Z:
					rec.normal = vec3(0, 0, 1);
					break;
				case NEG_Z:
					rec.normal = vec3(0, 0, -1);
					break;
				}
			}

			rec.obj = (hitable*)this;
			rec.mtrl = m_mtrl;

			// tangent coordinate.
			rec.du = normalize(getOrthoVector(rec.normal));
			rec.dv = normalize(cross(rec.normal, rec.du));

			rec.area = m_bbox.computeSurfaceArea();
		}

		return isHit;
	}

	bool cube::hit(
		const ray& r,
		const mat4& mtxL2W,
		real t_min, real t_max,
		hitrecord& rec) const
	{
		AT_ASSERT(false);	// Not support.
		return hit(r, t_min, t_max, rec);
	}

	aabb cube::getBoundingbox() const
	{
		return std::move(m_bbox);
	}

	vec3 cube::getRandomPosOn(sampler* sampler) const
	{
		vec3 pos;
		onGetRandomPosOn(pos, sampler);

		return std::move(pos);
	}

	cube::Face cube::onGetRandomPosOn(vec3& pos, sampler* sampler) const
	{
		auto r1 = sampler->nextSample();

		static const auto d = 1.0 / 6.0;
		real th = 0;

		Face face = Face::POS_X;

		for (int i = 0; i < 6; i++) {
			if (th <= r1 && r1 < th + d) {
				face = (Face)i;
				break;
			}

			th += d;
		}

		auto r2 = sampler->nextSample();
		auto r3 = sampler->nextSample();

		auto c = m_center;
		auto s = m_size * 0.5;

		vec3 leftbottom;
		vec3 lefttop;
		vec3 rightbottom;

		switch (face) {
		case POS_X:
			leftbottom = c + vec3(s.x, -s.y, s.z);
			lefttop = leftbottom + vec3(0, 2 * s.y, 0);
			rightbottom = leftbottom + vec3(0, 0, -2 * s.z);
			break;
		case NEG_X:
			leftbottom = c + vec3(-s.x, -s.y, s.z);
			lefttop = leftbottom + vec3(0, 2 * s.y, 0);
			rightbottom = leftbottom + vec3(0, 0, -2 * s.z);
			break;
		case POS_Y:
			leftbottom = c + vec3(-s.x, s.y, s.z);
			lefttop = leftbottom + vec3(0, 0, -2 * s.z);
			rightbottom = leftbottom + vec3(2 * s.x, 0, 0);
			break;
		case NEG_Y:
			leftbottom = c + vec3(-s.x, -s.y, s.z);
			lefttop = leftbottom + vec3(0, 0, -2 * s.z);
			rightbottom = leftbottom + vec3(2 * s.x, 0, 0);
			break;
		case POS_Z:
			leftbottom = c + vec3(s.x, -s.y, s.z);
			lefttop = leftbottom + vec3(0, 2 * s.y, 0);
			rightbottom = leftbottom + vec3(2 * s.x, 0, 0);
			break;
		case NEG_Z:
			leftbottom = c + vec3(s.x, -s.y, -s.z);
			lefttop = leftbottom + vec3(0, 2 * s.y, 0);
			rightbottom = leftbottom + vec3(2 * s.x, 0, 0);
			break;
		}

		vec3 x = rightbottom - leftbottom;
		vec3 y = lefttop - leftbottom;

		// For debug.
		vec3 nx = normalize(x);
		vec3 ny = normalize(y);

		pos = leftbottom + r2 * x + r3 * y;

		return face;
	}

	hitable::SamplingPosNormalPdf cube::getSamplePosNormalPdf(sampler* sampler) const
	{
		return getSamplePosNormalPdf(mat4::Identity, sampler);
	}

	hitable::SamplingPosNormalPdf cube::getSamplePosNormalPdf(
		const mat4& mtxL2W,
		sampler* sampler) const
	{
		vec3 pos;
		auto face = onGetRandomPosOn(pos, sampler);

		vec3 nml;
		switch (face) {
		case POS_X:
			nml = vec3(1, 0, 0);
			break;
		case NEG_X:
			nml = vec3(-1, 0, 0);
			break;
		case POS_Y:
			nml = vec3(0, 1, 0);
			break;
		case NEG_Y:
			nml = vec3(0, -1, 0);
			break;
		case POS_Z:
			nml = vec3(0, 0, 1);
			break;
		case NEG_Z:
			nml = vec3(0, 0, -1);
			break;
		}

		// TODO
		AT_ASSERT(false);
		auto area = 1;

		return std::move(hitable::SamplingPosNormalPdf(pos + nml * AT_MATH_EPSILON, nml, real(1) / area));
	}
}
