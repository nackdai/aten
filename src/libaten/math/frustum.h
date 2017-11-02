#pragma once

#include "defs.h"
#include "math/vec4.h"
#include "math/mat4.h"
#include "math/aabb.h"

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
			// normal.
			m_plane[0] = vec4(cross(dir[0], dir[1]), 0.0f);
			m_plane[1] = vec4(cross(dir[1], dir[2]), 0.0f);
			m_plane[2] = vec4(cross(dir[2], dir[3]), 0.0f);
			m_plane[3] = vec4(cross(dir[3], dir[0]), 0.0f);

			// offset.
			m_plane[0].w = dot(org, vec3(m_plane[0]));
			m_plane[1].w = dot(org, vec3(m_plane[1]));
			m_plane[2].w = dot(org, vec3(m_plane[2]));
			m_plane[3].w = dot(org, vec3(m_plane[3]));

			// points.
			m_points[0] = org;

			// TODO
			m_points[1] = org + dir[0] * real(100000.0);
			m_points[2] = org + dir[1] * real(100000.0);
			m_points[3] = org + dir[2] * real(100000.0);
			m_points[4] = org + dir[3] * real(100000.0);
		}

		enum Intersect {
			Miss,
			Inside,
			Intersecting,
		};

		Intersect intersect(const aabb& box) const
		{
			return intersect(box.minPos(), box.maxPos());
		}

		Intersect intersect(const vec3& minbox, const vec3& maxbox) const
		{
			// NOTE
			// http://www.txutxi.com/?p=584

			// TODO
			// SIMDfy...

			bool intersecting = false;

			vec3 b[] = { minbox, maxbox };

			for (int i = 0; i < AT_COUNTOF(m_plane); i++) {
				const vec4& plane = m_plane[i];

				int px = (plane.x > real(0.0));
				int py = (plane.y > real(0.0));
				int pz = (plane.z > real(0.0));

				// Dot product
				// project p-vertex on plane normal
				// (How far is p-vertex from the origin)
				real d = (plane.x * b[px].x)
					+ (plane.y * b[py].y)
					+ (plane.z * b[py].z);

				if (d < -plane.w) {
					return Intersect::Miss;
				}
				else {
					intersecting = true;
				}
			}

			// NOTE
			// https://cesium.com/blog/2017/02/02/tighter-frustum-culling-and-why-you-may-want-to-disregard-it/

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
					const auto& a = axis[3];
					
					auto axisLengthSquared = squared_length(a);

					int out1 = 0;
					int out2 = 0;

					for (int n = 0; n < AT_COUNTOF(diff); n++) {
						auto proj = dot(diff[i], a);

						if (proj >= axisLengthSquared) {
							out1++;
						}
						else if (proj < -axisLengthSquared) {
							out2++;
						}
					}

					if (out1 == AT_COUNTOF(diff) || out2 == AT_COUNTOF(diff)) {
						return Intersect::Miss;
					}

					intersecting |= (out1 != 0 || out2 != 0);
				}
			}

			return intersecting ? Intersect::Intersecting : Intersect::Inside;
		}

	private:
		// NOTE
		// Not use near, far plane.
		vec4 m_plane[4];

		// NOTE
		// 1 is origin, othere 4 are corner.
		vec4 m_points[5];
	};
}
