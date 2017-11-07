#pragma once

#include "defs.h"
#include "math/vec4.h"
#include "math/mat4.h"
#include "math/aabb.h"

#pragma optimize( "", off)


namespace aten {
	class frustum {
	public:
		frustum() {}
		~frustum() {}

	public:
		void update(
			const vec3& org,
			const vec3 dir[4])
		{
			// NOTE
			// 1--2
			// |  |
			// 0--3

			// normal.
			m_plane[0] = vec4(normalize(cross(dir[1], dir[0])), 0.0f);
			m_plane[1] = vec4(normalize(cross(dir[2], dir[1])), 0.0f);
			m_plane[2] = vec4(normalize(cross(dir[3], dir[2])), 0.0f);
			m_plane[3] = vec4(normalize(cross(dir[0], dir[3])), 0.0f);

			// offset.
			m_plane[0].w = dot(org, vec3(m_plane[0]));
			m_plane[1].w = dot(org, vec3(m_plane[1]));
			m_plane[2].w = dot(org, vec3(m_plane[2]));
			m_plane[3].w = dot(org, vec3(m_plane[3]));

			// points.
			m_points[0] = org;

			// TODO
			m_points[1] = org + dir[0] * real(10000.0);
			m_points[2] = org + dir[1] * real(10000.0);
			m_points[3] = org + dir[2] * real(10000.0);
			m_points[4] = org + dir[3] * real(10000.0);

			m_dir[0] = dir[0];
			m_dir[1] = dir[1];
			m_dir[2] = dir[2];
			m_dir[3] = dir[3];
		}

		void transform(const mat4& mtx)
		{
			vec3 org = m_points[0];

			org = mtx.apply(org);

			m_dir[0] = normalize(mtx.applyXYZ(m_dir[0]));
			m_dir[1] = normalize(mtx.applyXYZ(m_dir[1]));
			m_dir[2] = normalize(mtx.applyXYZ(m_dir[2]));
			m_dir[3] = normalize(mtx.applyXYZ(m_dir[3]));

			update(org, m_dir);
		}

		enum Intersect {
			Miss,
			Inside,
			Intersecting,
		};

		bool intersect(const aabb& box) const
		{
			return intersect(box.minPos(), box.maxPos());
		}

		bool intersect(const vec3& minbox, const vec3& maxbox) const
		{
			// NOTE
			// http://www.txutxi.com/?p=584
			// http://old.cescg.org/CESCG-2002/DSykoraJJelinek/
			// https://www.gamedev.net/forums/topic/672043-perfect-aabb-frustum-intersection-test/

			vec3 b[] = { minbox, maxbox };

			// Inside.
			bool intersecting = false;

			for (int i = 0; i < AT_COUNTOF(m_plane); i++) {
				const vec4& p = m_plane[i];

				int px = (p.x > real(0.0));
				int py = (p.y > real(0.0));
				int pz = (p.z > real(0.0));

				vec3 vmax(b[px].x, b[py].y, b[pz].z);
				vec3 vmin(b[1 - px].x, b[1 - py].y, b[1 - pz].z);

				auto d0 = dot((vec3)p, vmax);
				auto d1 = dot((vec3)p, vmin);

				if (d1 <= p.w && p.w <= d0) {
					// Intersecting.
					intersecting = true;
				}
				else if (d1 > p.w) {
					// Miss.
					return false;
				}
			}

			// NOTE
			// https://cesium.com/blog/2017/02/02/tighter-frustum-culling-and-why-you-may-want-to-disregard-it/

			// TODO
			// この方法では intersectingとinsideの違いを判定できないと思われる...

			if (intersecting) {
				auto center = (minbox + maxbox) * real(0.5);

				vec3 diff[] = {
					(vec3)m_points[0] - center,
					(vec3)m_points[1] - center,
					(vec3)m_points[2] - center,
					(vec3)m_points[3] - center,
					(vec3)m_points[4] - center,
				};

				// NOTE
				// AABBなので、x, y, z 軸に平行な軸になる.
				// ただし、正規化する必要がないので、大きさがAABBのハーフサイズになる.

				auto size = maxbox - minbox;
				vec3 axis[] = {
					vec3(size.x * real(0.5), real(0), real(0)),
					vec3(real(0), size.y * real(0.5), real(0)),
					vec3(real(0), real(0), size.z * real(0.5)),
				};

				for (int i = 0; i < 3; i++) {
					const auto& a = axis[i];
					
					auto axisLengthSquared = squared_length(a);

					int out1 = 0;
					int out2 = 0;

					if (axisLengthSquared == real(0)) {
						continue;
					}

					for (int n = 0; n < AT_COUNTOF(diff); n++) {
						auto proj = dot(diff[n], a);

						if (proj >= axisLengthSquared) {
							out1++;
						}
						else if (proj < -axisLengthSquared) {
							out2++;
						}
					}

					if (out1 == AT_COUNTOF(diff) || out2 == AT_COUNTOF(diff)) {
						//return Intersect::Miss;
						return false;
					}

					intersecting |= (out1 != 0 || out2 != 0);
				}
			}

			//return intersecting ? Intersect::Intersecting : Intersect::Inside;
			return true;
		}

	private:
		// NOTE
		// Not use near, far plane.
		vec4 m_plane[4];

		// NOTE
		// 1 is origin, othere 4 are corner.
		vec4 m_points[5];

		vec3 m_dir[4];
	};
}

#pragma optimize( "", on)
