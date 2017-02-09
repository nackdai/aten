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

		void add(primitive* s)
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

			for (size_t i = 0; i < m_objs.size(); i++) {
				auto o = m_objs[i];
				if (o->hit(r, t_min, t_max, tmp)) {
					if (tmp.t < rec.t) {
						rec = tmp;
						rec.obj = o;

						t_max = tmp.t;

						isHit = true;
					}
				}
			}

			return isHit;
		}

	private:
		std::vector<primitive*> m_objs;
	};

	class scene {
	public:
		scene() {}
		virtual ~scene() {}

		virtual void add(primitive* s) = 0;

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const = 0;

		void addLight(sphere* l)
		{
			m_lights.push_back(l);
		}

		uint32_t lightNum() const
		{
			return m_lights.size();
		}

		// TODO
		sphere* getLight(uint32_t i)
		{
			if (i >= lightNum()) {
				AT_ASSERT(false);
				return nullptr;
			}
			return m_lights[i];
		}

	private:
		// TODO
		std::vector<sphere*> m_lights;
	};

	template <typename ACCEL>
	class AcceledScene : public scene {
	public:
		AcceledScene() {}
		virtual ~AcceledScene() {}

	public:
		virtual void add(primitive* s) final
		{
			m_accel.add(s);
		}

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const final
		{
			auto isHit = m_accel.hit(r, t_min, t_max, rec);
			return isHit;
		}

	private:
		ACCEL m_accel;
	};
}
