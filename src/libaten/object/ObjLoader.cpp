#include <vector>
#include "tiny_obj_loader.h"
#include "defs.h"
#include "object/ObjLoader.h"
#include "object/object.h"

#include "scene/MaterialManager.h"
#include "material/lambert.h"

namespace aten
{
	object* ObjLoader::load(const char* path)
	{
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> mtrls;
		std::string err;

		// TODO
		// mtl_basepath

		auto flags = tinyobj::triangulation | tinyobj::calculate_normals;

		auto result = tinyobj::LoadObj(
			shapes, mtrls,
			err,
			path, nullptr,
			flags);
		AT_VRETURN(result, nullptr);

		object* obj = new object();

		vec3 shapemin(AT_MATH_INF);
		vec3 shapemax(-AT_MATH_INF);

		for (int p = 0; p < shapes.size(); p++) {
			const auto& shape = shapes[p];
			aten::shape* dstshape = new aten::shape();

			auto idxnum = shape.mesh.indices.size();
			auto vtxnum = shape.mesh.positions.size();

			vec3 pmin(AT_MATH_INF);
			vec3 pmax(-AT_MATH_INF);

			for (uint32_t i = 0; i < vtxnum; i += 3) {
				vertex v;

				v.pos.x = shape.mesh.positions[i + 0];
				v.pos.y = shape.mesh.positions[i + 1];
				v.pos.z = shape.mesh.positions[i + 2];

				v.nml.x = shape.mesh.normals[i + 0];
				v.nml.y = shape.mesh.normals[i + 1];
				v.nml.z = shape.mesh.normals[i + 2];

				pmin = vec3(
					std::min(pmin.x, v.pos.x),
					std::min(pmin.y, v.pos.y),
					std::min(pmin.z, v.pos.z));
				pmax = vec3(
					std::max(pmax.x, v.pos.x),
					std::max(pmax.y, v.pos.y),
					std::max(pmax.z, v.pos.z));

				dstshape->vertices.push_back(v);
			}

			shapemin = vec3(
				std::min(shapemin.x, pmin.x),
				std::min(shapemin.y, pmin.y),
				std::min(shapemin.z, pmin.z));
			shapemax = vec3(
				std::max(shapemax.x, pmax.x),
				std::max(shapemax.y, pmax.y),
				std::max(shapemax.z, pmax.z));

			for (uint32_t i = 0; i < idxnum; i += 3) {
				face* f = new face();

				f->idx[0] = shape.mesh.indices[i + 0];
				f->idx[1] = shape.mesh.indices[i + 1];
				f->idx[2] = shape.mesh.indices[i + 2];

				auto& v0 = dstshape->vertices[f->idx[0]];
				auto& v1 = dstshape->vertices[f->idx[1]];
				auto& v2 = dstshape->vertices[f->idx[2]];

				f->build(&v0, &v1, &v2);

				f->parent = dstshape;

				dstshape->faces.push_back(f);
			}

			// Assign material.
			auto mtrlidx = -1;
			if (shape.mesh.material_ids.size() > 0) {
				mtrlidx = shape.mesh.material_ids[0];
			}
			if (mtrlidx >= 0) {
				const auto mtrl = mtrls[mtrlidx];
				dstshape->mtrl = MaterialManager::get(mtrl.name);
			}
			if (!dstshape->mtrl){
				// dummy....
				AT_ASSERT(false);
				dstshape->mtrl = new lambert(vec3());
			}

			vtxnum = shape.mesh.texcoords.size();

			for (uint32_t i = 0; i < vtxnum; i += 2) {
				uint32_t vpos = i / 2;

				auto& v = dstshape->vertices[vpos];

				v.uv.x = shape.mesh.texcoords[i + 0];
				v.uv.y = shape.mesh.texcoords[i + 1];
			}

			dstshape->build();

			obj->m_shapes.push_back(dstshape);
		}

		obj->m_aabb.init(shapemin, shapemax);

		return obj;
	}
}
