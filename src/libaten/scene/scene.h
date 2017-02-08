#pragma once

#include <vector>
#include "primitive/sphere.h"

namespace aten {
	class accel {
	public:
		accel() {}
		virtual ~accel() {}

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const = 0;
	};

	class LinearList : public accel {
	public:
		LinearList() {}
		~LinearList() {}

		void add(sphere* s)
		{
			m_objs.push_back(s);
		}

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const final
		{
			bool isHit = false;

			hitrecord tmp;

			for (int i = 0; i < m_objs.size(); i++) {
				auto s = m_objs[i];
				if (s->hit(r, t_min, t_max, tmp)) {
					if (tmp.t < rec.t) {
						rec = tmp;

						t_max = tmp.t;
					}
				}
			}

			return isHit;
		}

	private:
		std::vector<sphere*> m_objs;
	};

	template <typename ACCEL>
	class scene {
	public:
		scene() {}
		~scene() {}

	public:
		void add(sphere* s)
		{
			m_accel.add(s);
		}

		bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const
		{
			auto isHit = m_accel.hit(r, t_min, t_max, rec);
			return isHit;
		}

	private:
		ACCEL m_accel;
	};
}
