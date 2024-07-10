#include <vector>

#include "tiny_obj_loader.h"

#include "ImageLoader.h"
#include "ObjLoader.h"
#include "utility.h"

//#pragma optimize( "", off)

namespace aten
{
    static std::string g_base;

    void ObjLoader::setBasePath(std::string_view base)
    {
        g_base = removeTailPathSeparator(base);
    }

    std::shared_ptr<aten::PolygonObject> ObjLoader::LoadFirstObj(
        std::string_view path,
        context& ctxt,
        aten::AssetManager& asset_manager,
        ObjLoader::FuncCreateMaterial callback_create_mtrl/*= nullptr*/,
        bool needComputeNormalOntime/*= false*/)
    {
        auto objs = load(
            path, ctxt, asset_manager, callback_create_mtrl, needComputeNormalOntime);

        return (!objs.empty() ? objs[0] : nullptr);
    }

    std::shared_ptr<aten::PolygonObject> ObjLoader::LoadFirstObjAndStoreToAssetManagerWithTag(
        std::string_view tag,
        std::string_view path,
        context& ctxt,
        aten::AssetManager& asset_manager,
        ObjLoader::FuncCreateMaterial callback_create_mtrl/*= nullptr*/,
        bool needComputeNormalOntime/*= false*/)
    {
        auto objs = LoadAndStoreToAssetManagerWithTag(
            tag, path, ctxt, asset_manager, callback_create_mtrl, needComputeNormalOntime);

        return (!objs.empty() ? objs[0] : nullptr);
    }

    std::vector<std::shared_ptr<aten::PolygonObject>> ObjLoader::load(
        std::string_view path,
        context& ctxt,
        aten::AssetManager& asset_manager,
        ObjLoader::FuncCreateMaterial callback_create_mtrl/*= nullptr*/,
        bool willSeparate/*= false*/,
        bool needComputeNormalOntime/*= false*/)
    {
        std::vector<std::shared_ptr<aten::PolygonObject>> objs;

        std::string pathname;
        std::string extname;
        std::string filename;

        getStringsFromPath(
            path,
            pathname,
            extname,
            filename);

        std::string fullpath(path);
        if (!g_base.empty()) {
            fullpath = g_base + "/" + fullpath;
        }

        return LoadAndStoreToAssetManagerWithTag(
            filename, fullpath, ctxt, asset_manager, callback_create_mtrl, willSeparate, needComputeNormalOntime);
    }

    std::vector<std::shared_ptr<aten::PolygonObject>> ObjLoader::LoadAndStoreToAssetManagerWithTag(
        std::string_view tag,
        std::string_view path,
        context& ctxt,
        aten::AssetManager& asset_manager,
        ObjLoader::FuncCreateMaterial callback_create_mtrl/*= nullptr*/,
        bool willSeparate/*= false*/,
        bool needComputeNormalOntime/*= false*/)
    {
        std::vector<std::shared_ptr<aten::PolygonObject>> objs;

        auto asset_obj = asset_manager.getObj(tag);
        if (asset_obj) {
            AT_PRINTF("There is same tag object. [%s]\n", tag.data());
            objs.push_back(asset_obj);
            return objs;
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
            path.data(), mtrlBasePath.c_str());

        if (!result) {
            AT_PRINTF("LoadObj Err[%s]\n", path.data());
            return objs;
        }

        auto obj(aten::TransformableFactory::createObject(ctxt));

        vec3 shapemin = vec3(AT_MATH_INF);
        vec3 shapemax = vec3(-AT_MATH_INF);

        uint32_t numPolygons = 0;

        for (int32_t p = 0; p < shapes.size(); p++) {
            const auto& shape = shapes[p];

            if (obj && obj->getName().empty()) {
                obj->setName(shape.name.c_str());
            }

            auto curVtxPos = ctxt.GetVertexNum();

            vec3 pmin = vec3(AT_MATH_INF);
            vec3 pmax = vec3(-AT_MATH_INF);

            auto face_num = shape.mesh.num_face_vertices.size();

            // Keep polygon counts.
            numPolygons += static_cast<uint32_t>(face_num);

            // TODO
            // Avoid duplicate.
            std::vector<tinyobj::index_t> vtx_info_list;

            std::vector<aten::TriangleParameter> face_parameters;

            // Aggregate vertex indices.
            for (uint32_t i = 0; i < face_num; i++) {
                // Loading as triangle is specified, so vertex num per triangle have to be 3.
                AT_ASSERT(shape.mesh.num_face_vertices[i] == 3);

                aten::TriangleParameter faceParam;

                const auto& idx_0 = shape.mesh.indices[i * 3 + 0];
                const auto& idx_1 = shape.mesh.indices[i * 3 + 1];
                const auto& idx_2 = shape.mesh.indices[i * 3 + 2];

                vtx_info_list.push_back(idx_0);
                auto it = vtx_info_list.back();
                faceParam.idx[0] = static_cast<uint32_t>(vtx_info_list.size()) - 1 + curVtxPos;

                vtx_info_list.push_back(idx_1);
                it = vtx_info_list.back();
                faceParam.idx[1] = static_cast<uint32_t>(vtx_info_list.size()) - 1 + curVtxPos;

                vtx_info_list.push_back(idx_2);
                it = vtx_info_list.back();
                faceParam.idx[2] = static_cast<uint32_t>(vtx_info_list.size()) - 1 + curVtxPos;

                face_parameters.push_back(faceParam);
            }

            // Vertices.
            for (const auto& idx : vtx_info_list) {
                vertex vtx;

                vtx.pos.x = attrib.vertices[idx.vertex_index * 3 + 0];
                vtx.pos.y = attrib.vertices[idx.vertex_index * 3 + 1];
                vtx.pos.z = attrib.vertices[idx.vertex_index * 3 + 2];
                vtx.pos.w = float(0);

                // NOTE:
                // If vtx.uv.z == 1, normal will be computed during rendering.

                // TODO:
                // Even if 1.0 is applied to uv.z, it might be overwritten with -1 at L210.

                if (idx.normal_index < 0) {
                    vtx.uv.z = float(1);
                }
                else {
                    vtx.nml.x = attrib.normals[idx.normal_index * 3 + 0];
                    vtx.nml.y = attrib.normals[idx.normal_index * 3 + 1];
                    vtx.nml.z = attrib.normals[idx.normal_index * 3 + 2];
                    vtx.uv.z = needComputeNormalOntime ? float(1) : float(0);
                }

                if (std::isnan(vtx.nml.x) || std::isnan(vtx.nml.y) || std::isnan(vtx.nml.z))
                {
                    // If one of normal elements is NaN, normal will be computed during rendering.
                    vtx.nml = aten::vec4(float(0), float(1), float(0), 1);
                }

                if (idx.texcoord_index >= 0) {
                    vtx.uv.x = attrib.texcoords[idx.texcoord_index * 2 + 0];
                    vtx.uv.y = attrib.texcoords[idx.texcoord_index * 2 + 1];
                }
                else {
                    // Specify not have texture coordinates.
                    vtx.uv.z = float(-1);
                }

                ctxt.AddVertex(vtx);

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

            std::shared_ptr<aten::TriangleGroupMesh> dst_shape;
            int32_t prev_mtrl_idx = -1;

            // One shape has one material.It means another shape would be created if different material appear.
            for (uint32_t i = 0; i < face_num; i++) {
                // Loading as triangle is specified, so vertex num per triangle have to be 3.
                AT_ASSERT(shape.mesh.num_face_vertices[i] == 3);

                int32_t m = shape.mesh.material_ids[i];

                if (m < 0 && !dst_shape) {
                    // If a material doesn't exist.
                    dst_shape = std::make_shared<aten::TriangleGroupMesh>();

                    auto regireterd_mtrl = asset_manager.getMtrlByIdx(0);
                    if (!regireterd_mtrl && callback_create_mtrl) {
                        regireterd_mtrl = callback_create_mtrl(
                            "", ctxt, MaterialType::Lambert, aten::vec3(1), "", "");
                    }

                    AT_ASSERT(regireterd_mtrl);

                    dst_shape->SetMaterial(regireterd_mtrl);
                }
                else if (prev_mtrl_idx != m) {
                    // If different material appear.

                    if (dst_shape) {
                        // If the shape already exist.

                        auto mtrl = dst_shape->GetMaterial();

                        if (mtrl->param().type == aten::MaterialType::Emissive) {
                            // Export the object which has an emissive material as the emissive object.
                            auto emitobj(aten::TransformableFactory::createObject(ctxt));
                            emitobj->appendShape(dst_shape);
                            emitobj->setBoundingBox(aten::aabb(pmin, pmax));
                            objs.push_back(std::move(emitobj));
                        }
                        else {
                            // When different material appear, register the shape to the object.
                            // And, create new shape and shift to it later.
                            obj->appendShape(dst_shape);
                        }
                    }

                    // Create new shape for new material.
                    dst_shape = std::make_shared<aten::TriangleGroupMesh>();
                    prev_mtrl_idx = m;

                    if (prev_mtrl_idx >= 0) {
                        // Apply new materil to the shape.
                        const auto& mtrl = mtrls[prev_mtrl_idx];

                        auto aten_mtrl = asset_manager.getMtrl(mtrl.name);

                        if (!aten_mtrl && callback_create_mtrl) {
                            std::shared_ptr<material> new_mtrl(
                                callback_create_mtrl(
                                    mtrl.name,
                                    ctxt,
                                    MaterialType::Lambert,
                                    aten::vec3(mtrl.diffuse[0], mtrl.diffuse[1], mtrl.diffuse[2]),
                                    mtrl.diffuse_texname,
                                    mtrl.bump_texname));
                            asset_manager.registerMtrl(mtrl.name, new_mtrl);
                            dst_shape->SetMaterial(new_mtrl);
                        }
                        else {
                            dst_shape->SetMaterial(aten_mtrl);
                        }
                    }

                    if (!dst_shape->GetMaterial()) {
                        // If the shape doesn't have a material until here, set dummy material....

                        // Only lambertian.
                        const auto& objmtrl = mtrls[m];

                        std::shared_ptr<aten::material> mtrl;

                        if (callback_create_mtrl) {
                            mtrl = callback_create_mtrl(
                                    objmtrl.name,
                                    ctxt,
                                    MaterialType::Lambert,
                                    aten::vec3(objmtrl.diffuse[0], objmtrl.diffuse[1], objmtrl.diffuse[2]),
                                    objmtrl.diffuse_texname,
                                    objmtrl.bump_texname);
                        }
                        else {
                            aten::vec3 diffuse(objmtrl.diffuse[0], objmtrl.diffuse[1], objmtrl.diffuse[2]);

                            aten::texture* albedoMap = nullptr;
                            aten::texture* normalMap = nullptr;

                            // Albedo map.
                            if (!objmtrl.diffuse_texname.empty()) {
                                auto tex = asset_manager.getTex(objmtrl.diffuse_texname.c_str());

                                if (tex) {
                                    albedoMap = tex.get();
                                }
                                else {
                                    std::string texname = pathname + "/" + objmtrl.diffuse_texname;
                                    auto loaded_img = aten::ImageLoader::load(texname, ctxt, asset_manager);
                                    albedoMap = loaded_img.get();
                                }
                            }

                            // Normal map.
                            if (!objmtrl.bump_texname.empty()) {
                                auto tex = asset_manager.getTex(objmtrl.bump_texname.c_str());

                                if (tex) {
                                    normalMap = tex.get();
                                }
                                else {
                                    std::string texname = pathname + "/" + objmtrl.bump_texname;
                                    auto loaded_img = aten::ImageLoader::load(texname, ctxt, asset_manager);
                                    normalMap = loaded_img.get();
                                }
                            }

                            aten::MaterialParameter mtrlParam;
                            mtrlParam.type = aten::MaterialType::Lambert;
                            mtrlParam.baseColor = diffuse;

                            mtrl = ctxt.CreateMaterialWithMaterialParameter(
                                mtrlParam,
                                albedoMap,
                                normalMap,
                                nullptr);
                        }

                        mtrl->setName(objmtrl.name.c_str());

                        dst_shape->SetMaterial(mtrl);

                        asset_manager.registerMtrl(mtrl->name(), mtrl);
                    }
                }

                auto& face_param = face_parameters[i];

                const auto& v0 = ctxt.GetVertex(face_param.idx[0]);
                const auto& v1 = ctxt.GetVertex(face_param.idx[1]);
                const auto& v2 = ctxt.GetVertex(face_param.idx[2]);

                if (v0.uv.z == float(1)
                    || v1.uv.z == float(1)
                    || v2.uv.z == float(1)
                    || needComputeNormalOntime)
                {
                    face_param.needNormal = 1;
                }

                face_param.mtrlid = dst_shape->GetMaterial()->id();
                face_param.mesh_id = dst_shape->get_mesh_id();

                auto f = ctxt.CreateTriangle(face_param);

                dst_shape->AddFace(f);
            }

            // Register the shape to the object.
            {
                const auto& mtrl = dst_shape->GetMaterial();

                if (willSeparate) {
                    // Split the object for each shape.
                    obj->appendShape(dst_shape);
                    obj->setBoundingBox(aten::aabb(pmin, pmax));
                    objs.push_back(obj);

                    if (p + 1 < shapes.size()) {
                        obj = aten::TransformableFactory::createObject(ctxt);
                    }
                    else {
                        // NOTE
                        // https://stackoverflow.com/questions/16151550/c11-when-clearing-shared-ptr-should-i-use-reset-or-set-to-nullptr
                        obj = nullptr;
                    }
                }
                else if (mtrl->param().type == aten::MaterialType::Emissive) {
                    // Export the object which has an emissive material as the emissive object.
                    auto emitobj(aten::TransformableFactory::createObject(ctxt));
                    emitobj->appendShape(dst_shape);
                    emitobj->setBoundingBox(aten::aabb(pmin, pmax));
                    objs.push_back(std::move(emitobj));
                }
                else {
                    obj->appendShape(dst_shape);
                }
            }
        }

        if (!willSeparate) {
            AT_ASSERT(obj);

            obj->setBoundingBox(aten::aabb(shapemin, shapemax));

            // TODO
            asset_manager.registerObj(tag, obj);

            objs.push_back(std::move(obj));
        }

        auto vtxNum = ctxt.GetVertexNum();

        AT_PRINTF("(%s)\n", path.data());
        AT_PRINTF("    %d[vertices]\n", vtxNum);
        AT_PRINTF("    %d[polygons]\n", numPolygons);

        return objs;
    }
}
