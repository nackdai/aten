#include <vector>
#include "tiny_obj_loader.h"
#include "ObjLoader.h"
#include "AssetManager.h"
#include "utility.h"
#include "ImageLoader.h"

//#pragma optimize( "", off)

namespace aten
{
    static std::string g_base;

    void ObjLoader::setBasePath(const std::string& base)
    {
        g_base = removeTailPathSeparator(base);
    }

    object* ObjLoader::load(
        const std::string& path,
        context& ctxt,
        bool needComputeNormalOntime/*= false*/)
    {
        std::vector<object*> objs;
        load(objs, path, ctxt, needComputeNormalOntime);

        object* ret = (!objs.empty() ? objs[0] : nullptr);
        return ret;
    }

    object* ObjLoader::load(
        const std::string& tag,
        const std::string& path,
        context& ctxt,
        bool needComputeNormalOntime/*= false*/)
    {
        std::vector<object*> objs;
        load(objs, tag, path, ctxt, needComputeNormalOntime);

        object* ret = (!objs.empty() ? objs[0] : nullptr);
        return ret;
    }

    void ObjLoader::load(
        std::vector<object*>& objs,
        const std::string& path,
        context& ctxt,
        bool willSeparate/*= false*/,
        bool needComputeNormalOntime/*= false*/)
    {
        std::string pathname;
        std::string extname;
        std::string filename;

        getStringsFromPath(
            path,
            pathname,
            extname,
            filename);

        std::string fullpath = path;
        if (!g_base.empty()) {
            fullpath = g_base + "/" + fullpath;
        }

        load(objs, filename, fullpath, ctxt, willSeparate, needComputeNormalOntime);
    }

    void ObjLoader::load(
        std::vector<object*>& objs,
        const std::string& tag,
        const std::string& path,
        context& ctxt,
        bool willSeparate/*= false*/,
        bool needComputeNormalOntime/*= false*/)
    {
        object* obj = AssetManager::getObj(tag);
        if (obj) {
            AT_PRINTF("There is same tag object. [%s]\n", tag.c_str());
            objs.push_back(obj);
            return;
        }

        std::string pathname;
        std::string extname;
        std::string filename;

        aten::getStringsFromPath(path, pathname, extname, filename);

        std::string mtrlBasePath = pathname + "/";

        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> mtrls;
        std::string err, warn;

        auto result = tinyobj::LoadObj(
            &attrib, &shapes, &mtrls,
            &warn, &err,
            path.c_str(), mtrlBasePath.c_str());

        if (!result) {
            AT_PRINTF("LoadObj Err[%s]\n", path.c_str());
            return;
        }

        obj = aten::TransformableFactory::createObject(ctxt);

        vec3 shapemin = vec3(AT_MATH_INF);
        vec3 shapemax = vec3(-AT_MATH_INF);

        uint32_t numPolygons = 0;

        for (int p = 0; p < shapes.size(); p++) {
            const auto& shape = shapes[p];

            //AT_ASSERT(shape.mesh.positions.size() == shape.mesh.normals.size());
            //AT_ASSERT(shape.mesh.positions.size() / 3 == shape.mesh.texcoords.size() / 2);

            auto curVtxPos = ctxt.getVertexNum();

            vec3 pmin = vec3(AT_MATH_INF);
            vec3 pmax = vec3(-AT_MATH_INF);

            auto face_num = shape.mesh.num_face_vertices.size();

            // TODO
            // Avoid duplicate.
            std::vector<tinyobj::index_t> vtx_info_list;

            std::vector<aten::PrimitiveParamter> face_parameters;

            // Aggregate vertex indices.
            for (uint32_t i = 0; i < face_num; i++) {
                // Loading as triangle is specified, so vertex num per face have to be 3.
                AT_ASSERT(shape.mesh.num_face_vertices[i] == 3);

                aten::PrimitiveParamter faceParam;

                const auto& idx_0 = shape.mesh.indices[i * 3 + 0];
                const auto& idx_1 = shape.mesh.indices[i * 3 + 1];
                const auto& idx_2 = shape.mesh.indices[i * 3 + 2];

                vtx_info_list.push_back(idx_0);
                auto it = vtx_info_list.back();
                faceParam.idx[0] = vtx_info_list.size() - 1 + curVtxPos;

                vtx_info_list.push_back(idx_1);
                it = vtx_info_list.back();
                faceParam.idx[1] = vtx_info_list.size() - 1 + curVtxPos;

                vtx_info_list.push_back(idx_2);
                it = vtx_info_list.back();
                faceParam.idx[2] = vtx_info_list.size() - 1 + curVtxPos;

                face_parameters.push_back(faceParam);
            }

            // Vertices.
            for (const auto& idx : vtx_info_list) {
                vertex vtx;

                vtx.pos.x = attrib.vertices[idx.vertex_index * 3 + 0];
                vtx.pos.y = attrib.vertices[idx.vertex_index * 3 + 1];
                vtx.pos.z = attrib.vertices[idx.vertex_index * 3 + 2];
                vtx.pos.w = real(0);

                // NOTE
                // If vtx.uv.z == 1, normal will be computed during rendering.

                if (idx.normal_index < 0) {
                    // Flag not to specify normal.
                    vtx.uv.z = real(1);
                }
                else {
                    vtx.nml.x = attrib.normals[idx.normal_index * 3 + 0];
                    vtx.nml.y = attrib.normals[idx.normal_index * 3 + 1];
                    vtx.nml.z = attrib.normals[idx.normal_index * 3 + 2];
                    vtx.uv.z = needComputeNormalOntime ? real(1) : real(0);
                }

                if (std::isnan(vtx.nml.x) || std::isnan(vtx.nml.y) || std::isnan(vtx.nml.z))
                {
                    // If one of normal elements is NaN, normal will be computed during rendering.
                    vtx.nml = aten::vec4(real(0), real(1), real(0), 1);
                }

                if (idx.texcoord_index >= 0) {
                    vtx.uv.x = attrib.texcoords[idx.texcoord_index * 2 + 0];
                    vtx.uv.y = attrib.texcoords[idx.texcoord_index * 2 + 1];
                }
                else {
                    // Specify not have texture coordinates.
                    vtx.uv.z = real(-1);
                }

                ctxt.addVertex(vtx);

                // To compute bouding box, store min/max position.
                pmin = vec3(
                    std::min(pmin.x, vtx.pos.x),
                    std::min(pmin.y, vtx.pos.y),
                    std::min(pmin.z, vtx.pos.z));
                pmax = vec3(
                    std::max(pmax.x, vtx.pos.x),
                    std::max(pmax.y, vtx.pos.y),
                    std::max(pmax.z, vtx.pos.z));
            }

            // Compute bounding box.
            shapemin = vec3(
                std::min(shapemin.x, pmin.x),
                std::min(shapemin.y, pmin.y),
                std::min(shapemin.z, pmin.z));
            shapemax = vec3(
                std::max(shapemax.x, pmax.x),
                std::max(shapemax.y, pmax.y),
                std::max(shapemax.z, pmax.z));

            aten::objshape* dst_shape = nullptr;
            int prev_mtrl_idx = -1;

            // One shape has one material.It means another shape would be created if different material appear.
            for (uint32_t i = 0; i < face_num; i++) {
                // Loading as triangle is specified, so vertex num per face have to be 3.
                AT_ASSERT(shape.mesh.num_face_vertices[i] == 3);

                int m = shape.mesh.material_ids[i];

                if (m < 0 && !dst_shape) {
                    // If a material doesn't exist.
                    dst_shape = new aten::objshape();
                    dst_shape->setMaterial(AssetManager::getMtrlByIdx(0));
                }
                else if (prev_mtrl_idx != m) {
                    // If different material appear.

                    if (dst_shape) {
                        // If the shape already exist.

                        auto mtrl = dst_shape->getMaterial();

                        if (willSeparate) {
                            // Split the object if it has different materials
                            obj->appendShape(dst_shape);
                            obj->setBoundingBox(aten::aabb(pmin, pmax));
                            objs.push_back(obj);

                            // Create new object for next steps after here.
                            obj = aten::TransformableFactory::createObject(ctxt);
                        }
                        if (mtrl->param().type == aten::MaterialType::Emissive) {
                            // Export the object which has an emissive material as the emissive object.
                            auto emitobj = aten::TransformableFactory::createObject(ctxt);
                            emitobj->appendShape(dst_shape);
                            emitobj->setBoundingBox(aten::aabb(pmin, pmax));
                            objs.push_back(emitobj);
                        }
                        else {
                            // When different material appear, register the shape to the object.
                            // And, create new shape and shift to it later.
                            obj->appendShape(dst_shape);
                        }
                    }

                    // Create new shape for new material.
                    dst_shape = new aten::objshape();
                    prev_mtrl_idx = m;

                    if (prev_mtrl_idx >= 0) {
                        // Apply new materil to the shape.
                        const auto& mtrl = mtrls[prev_mtrl_idx];
                        dst_shape->setMaterial(AssetManager::getMtrl(mtrl.name));
                    }

                    if (!dst_shape->getMaterial()) {
                        // If the shape doesn't have a material until here, set dummy material....

                        // Only lambertian.
                        const auto& objmtrl = mtrls[m];

                        aten::vec3 diffuse(objmtrl.diffuse[0], objmtrl.diffuse[1], objmtrl.diffuse[2]);

                        aten::texture* albedoMap = nullptr;
                        aten::texture* normalMap = nullptr;

                        // Albedo map.
                        if (!objmtrl.diffuse_texname.empty()) {
                            albedoMap = AssetManager::getTex(objmtrl.diffuse_texname.c_str());

                            if (!albedoMap) {
                                std::string texname = pathname + "/" + objmtrl.diffuse_texname;
                                albedoMap = aten::ImageLoader::load(texname, ctxt);
                            }
                        }

                        // Normal map.
                        if (!objmtrl.bump_texname.empty()) {
                            normalMap = AssetManager::getTex(objmtrl.bump_texname.c_str());

                            if (!normalMap) {
                                std::string texname = pathname + "/" + objmtrl.bump_texname;
                                normalMap = aten::ImageLoader::load(texname, ctxt);
                            }
                        }

                        aten::MaterialParameter mtrlParam;
                        mtrlParam.baseColor = diffuse;

                        aten::material* mtrl = ctxt.createMaterialWithMaterialParameter(
                            aten::MaterialType::Lambert,
                            mtrlParam,
                            albedoMap,
                            normalMap,
                            nullptr);

                        mtrl->setName(objmtrl.name.c_str());

                        dst_shape->setMaterial(mtrl);

                        AssetManager::registerMtrl(mtrl->name(), mtrl);
                    }
                }

                auto& face_param = face_parameters[i];

                auto& v0 = ctxt.getVertex(face_param.idx[0]);
                auto& v1 = ctxt.getVertex(face_param.idx[1]);
                auto& v2 = ctxt.getVertex(face_param.idx[2]);

                if (v0.uv.z == real(1)
                    || v1.uv.z == real(1)
                    || v2.uv.z == real(1))
                {
                    face_param.needNormal = 1;
                }

                face_param.mtrlid = dst_shape->getMaterial()->id();
                face_param.gemoid = dst_shape->getGeomId();

                auto f = ctxt.createTriangle(face_param);

                dst_shape->addFace(f);
            }

            // Keep polygon counts.
            numPolygons++;

            // Register the shape to the object.
            {
                const auto& mtrl = dst_shape->getMaterial();

                if (willSeparate) {
                    // Split the object for each shape.
                    obj->appendShape(dst_shape);
                    obj->setBoundingBox(aten::aabb(pmin, pmax));
                    objs.push_back(obj);

                    if (p + 1 < shapes.size()) {
                        obj = aten::TransformableFactory::createObject(ctxt);
                    }
                    else {
                        obj = nullptr;
                    }
                }
                else if (mtrl->param().type == aten::MaterialType::Emissive) {
                    // Export the object which has an emissive material as the emissive object.
                    auto emitobj = aten::TransformableFactory::createObject(ctxt);
                    emitobj->appendShape(dst_shape);
                    emitobj->setBoundingBox(aten::aabb(pmin, pmax));
                    objs.push_back(emitobj);
                }
                else {
                    obj->appendShape(dst_shape);
                }
            }
        }

        if (!willSeparate) {
            AT_ASSERT(obj);

            obj->setBoundingBox(aten::aabb(shapemin, shapemax));
            objs.push_back(obj);

            // TODO
            AssetManager::registerObj(tag, obj);
        }

        auto vtxNum = ctxt.getVertexNum();

        AT_PRINTF("(%s)\n", path.c_str());
        AT_PRINTF("    %d[vertices]\n", vtxNum);
        AT_PRINTF("    %d[polygons]\n", numPolygons);
    }
}
