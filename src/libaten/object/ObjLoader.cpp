#include <vector>
#include "tiny_obj_loader.h"
#include "defs.h"
#include "object/ObjLoader.h"
#include "object/object.h"

// TODO
#include "material/diffuse.h"

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

			for (uint32_t i = 0; i < idxnum; i += 3) {
				face f;

				f.idx[0] = shape.mesh.indices[i + 0];
				f.idx[1] = shape.mesh.indices[i + 1];
				f.idx[2] = shape.mesh.indices[i + 2];

				dstshape->faces.push_back(f);
			}

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
					min(pmin.x, v.pos.x),
					min(pmin.y, v.pos.y),
					min(pmin.z, v.pos.z));
				pmax = vec3(
					max(pmax.x, v.pos.x),
					max(pmax.y, v.pos.y),
					max(pmax.z, v.pos.z));

				dstshape->vertices.push_back(v);
			}

			dstshape->bbox.init(pmin, pmax);

			shapemin = vec3(
				min(shapemin.x, pmin.x),
				min(shapemin.y, pmin.y),
				min(shapemin.z, pmin.z));
			shapemax = vec3(
				max(shapemax.x, pmax.x),
				max(shapemax.y, pmax.y),
				max(shapemax.z, pmax.z));

			// TODO
			dstshape->mtrl = new diffuse(vec3(0.75, 0, 0));

			vtxnum = shape.mesh.texcoords.size();

			for (uint32_t i = 0; i < vtxnum; i += 2) {
				uint32_t vpos = i / 2;

				auto& v = dstshape->vertices[vpos];

				v.uv[0] = shape.mesh.texcoords[i + 0];
				v.uv[1] = shape.mesh.texcoords[i + 1];
			}

			obj->m_shapes.push_back(dstshape);
		}

		obj->m_aabb.init(shapemin, shapemax);

		return obj;
	}
}
