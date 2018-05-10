#pragma once 

#include <vector>
#include <functional>

#include "types.h"
#include "math/aabb.h"
#include "math/vec3.h"
#include "material/material.h"
#include "sampler/sampler.h"

//#define ENABLE_TANGENTCOORD_IN_HITREC

namespace aten {
	class hitable;
	class accelerator;

	struct hitrecord {
		vec3 p;

		vec3 normal;

#ifdef ENABLE_TANGENTCOORD_IN_HITREC
		// tangent coordinate.
		vec3 du;
		vec3 dv;
#endif

		union {
			// For triangle.
			struct {
				// texture coordinate.
				real u;
				real v;
				real padding;
			};
			// For voxel.
			vec3 albedo;
		};

		real area{ real(1) };

		union {
			int mtrlid;

			// For voxel.
			struct {
				int unused : 31;
				int isVoxel : 1;
			};
		};

		AT_DEVICE_API hitrecord()
		{
			u = v = real(0);
			mtrlid = -1;
		}
	};

	struct Intersection {
		real t{ AT_MATH_INF };

		int objid{ -1 };

		union {
			// For triangle.
			struct {
				// for cube.
				short face;

				short mtrlid;

				int meshid;
			};
			// For voxel.
			struct {
				struct {
					uint32_t area : 31;
					uint32_t isVoxel : 1;
				};
				real clr_r;
			};
		};

		union {
			// For triangle.
			struct {
				int primid;
				real a, b;	// barycentric
				real padding;
			};
			// Fox voxel.
			struct {
				real nml_x;
				real nml_y;
				union {
					uint32_t signNmlZ : 1;
					uint32_t clr_g : 31;
				};
				real clr_b;
			};
		};

		AT_DEVICE_API Intersection()
		{
			mtrlid = -1;
			meshid = -1;
			primid = -1;
		}
	};

	using NotifyChanged = std::function<void(hitable*)>;

	class hitable {
	public:
		hitable(const char* name = nullptr)
		{
			if (name) {
				m_name = name;
			}
		}
		virtual ~hitable() {}

	public:
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const = 0;

		virtual const aabb& getBoundingbox() const
		{
			return m_aabb;
		}
		void setBoundingBox(const aabb& bbox)
		{
			m_aabb = bbox;
		}

		virtual aabb getTransformedBoundingBox() const
		{
			return std::move(m_aabb);
		}

		virtual const hitable* getHasObject() const
		{
			return nullptr;
		}

		virtual const hitable* getHasSecondObject() const
		{
			return nullptr;
		}

		bool isInstance() const
		{
			return (getHasObject() != nullptr);
		}

		virtual int geomid() const
		{
			return -1;
		}

		virtual accelerator* getInternalAccelerator();

		struct SamplePosNormalPdfResult {
			aten::vec3 pos;
			aten::vec3 nml;
			real area;

			real a;
			real b;
			int primid{ -1 };
		};

		virtual void getSamplePosNormalArea(SamplePosNormalPdfResult* result, sampler* sampler) const
		{
			AT_ASSERT(false);
		}

		static void evalHitResult(
			const hitable* obj,
			const ray& r,
			hitrecord& rec,
			const Intersection& isect)
		{
			if (isect.isVoxel) {
				// For voxel.
				rec.area = expandTo32bitFloat(isect.area);

				// Repair normal.
				auto nml_z = aten::sqrt(std::min<real>(real(1) - isect.nml_x * isect.nml_x + isect.nml_y * isect.nml_y, real(1)));
				nml_z *= isect.signNmlZ ? real(-1) : real(1);
				rec.normal = normalize(vec3(isect.nml_x, isect.nml_y, nml_z));

				// Compute hit point.
				rec.p = r.org + isect.t * r.dir;
				rec.p = rec.p + AT_MATH_EPSILON * rec.normal;

				// Repair Albedo color.
				rec.albedo.x = isect.clr_r;
				rec.albedo.y = expandTo32bitFloat(isect.clr_g);
				rec.albedo.z = isect.clr_b;

				// Flag if voxel or not.
				rec.isVoxel = true;
			}
			else {
				obj->evalHitResult(r, rec, isect);
				rec.mtrlid = isect.mtrlid;

				AT_ASSERT(!rec.isVoxel);
			}

#ifdef ENABLE_TANGENTCOORD_IN_HITREC
			// tangent coordinate.
			rec.du = normalize(getOrthoVector(rec.normal));
			rec.dv = normalize(cross(rec.normal, rec.du));
#endif
		}

		static void evalHitResultForAreaLight(
			const hitable* obj,
			const ray& r,
			hitrecord& rec,
			const Intersection& isect)
		{
			obj->evalHitResult(r, rec, isect);
			rec.mtrlid = isect.mtrlid;

#ifdef ENABLE_TANGENTCOORD_IN_HITREC
			// tangent coordinate.
			rec.du = normalize(getOrthoVector(rec.normal));
			rec.dv = normalize(cross(rec.normal, rec.du));
#endif
		}

		using FuncPreDraw = std::function<void(const aten::mat4& mtxL2W, const aten::mat4& mtxPrevL2W, int parentId, int basePrimId)>;

		virtual void draw(
			FuncPreDraw func,
			const aten::mat4& mtxL2W,
			const aten::mat4& mtxPrevL2W,
			int parentId,
			uint32_t triOffset)
		{
			// For rasterize rendering.
			AT_ASSERT(false);
		}

		using FuncDrawAABB = std::function<void(const aten::mat4&)>;

		virtual void drawAABB(
			FuncDrawAABB func,
			const aten::mat4& mtxL2W)
		{
			// For debug rendering.
			AT_ASSERT(false);
		}

		void setFuncNotifyChanged(NotifyChanged onNotifyChanged)
		{
			m_onNotifyChanged = onNotifyChanged;
		}

		virtual void update()
		{
			// Nothing is done...
		}

		virtual uint32_t getTriangleCount() const
		{
			return 0;
		}

		void setName(const char* name)
		{
			m_name = name;
		}
		const char* name()
		{
			return m_name.c_str();
		}

	protected:
		void onNotifyChanged()
		{
			if (m_onNotifyChanged) {
				m_onNotifyChanged(this);
			}
		}

	private:
		virtual void evalHitResult(
			const ray& r,
			hitrecord& rec,
			const Intersection& isect) const
		{
			AT_ASSERT(false);
		}

	private:
		std::string m_name;
		aabb m_aabb;

		NotifyChanged m_onNotifyChanged;
	};
}
