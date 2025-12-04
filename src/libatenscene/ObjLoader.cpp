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
        ObjLoader::FuncCreateMaterial callback_create_mtrl/*= nullptr*/,
        FuncFindMaterialFromSceneContextByName callback_find_mtrl/*= nullptr*/,
        bool need_compute_normal_on_the_fly/*= false*/)
    {
        auto objs = Load(
            path, ctxt,
            callback_create_mtrl,
            callback_find_mtrl,
            need_compute_normal_on_the_fly);

        return (!objs.empty() ? objs[0] : nullptr);
    }

    std::vector<std::shared_ptr<aten::PolygonObject>> ObjLoader::Load(
        std::string_view path,
        context& ctxt,
        ObjLoader::FuncCreateMaterial callback_create_mtrl/*= nullptr*/,
        FuncFindMaterialFromSceneContextByName callback_find_mtrl/*= nullptr*/,
        bool will_register_shape_as_separate_obj/*= false*/,
        bool need_compute_normal_on_the_fly/*= false*/)
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

        return OnLoad(
            fullpath, ctxt,
            callback_create_mtrl,
            callback_find_mtrl,
            will_register_shape_as_separate_obj,
            need_compute_normal_on_the_fly);
    }

    std::vector<std::shared_ptr<aten::PolygonObject>> ObjLoader::OnLoad(
        std::string_view path,
        context& ctxt,
        ObjLoader::FuncCreateMaterial callback_create_mtrl/*= nullptr*/,
        FuncFindMaterialFromSceneContextByName callback_find_mtrl/*= nullptr*/,
        bool will_register_shape_as_separate_obj/*= false*/,
        bool need_compute_normal_on_the_fly/*= false*/)
    {
        std::vector<std::shared_ptr<aten::PolygonObject>> objs;

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

        // NOTE:
        // obj: Entire obj file.
        // shape: Polygon group in obj file. This is grouped in each material.

        std::shared_ptr<aten::PolygonObject> obj;

        auto create_obj_functor = [](
            std::shared_ptr<aten::PolygonObject>& obj,
            const tinyobj::shape_t& shape,
            aten::context& ctxt) {
            if (!obj) {
                obj = aten::TransformableFactory::createObject(ctxt);
            }
            if (obj->getName().empty()) {
                obj->setName(shape.name);
            }
        };

        vec3 shape_bbox_min = vec3(AT_MATH_INF);
        vec3 shape_bbox_max = vec3(-AT_MATH_INF);

        uint32_t all_triangle_num = 0;

        for (int32_t p = 0; p < shapes.size(); p++) {
            const auto& shape = shapes[p];

            auto curr_vtx_pos = ctxt.GetVertexNum();

            auto triangle_num = shape.mesh.num_face_vertices.size();

            // Keep polygon counts.
            all_triangle_num += static_cast<uint32_t>(triangle_num);

            // TODO
            // Avoid duplicate.
            std::vector<tinyobj::index_t> vtx_info_list;

            std::vector<aten::TriangleParameter> triangle_parameters;

            // Aggregate vertex indices.
            for (uint32_t i = 0; i < triangle_num; i++) {
                // Loading as triangle is specified, so vertex num per triangle have to be 3.
                AT_ASSERT(shape.mesh.num_face_vertices[i] == 3);

                aten::TriangleParameter faceParam;

                const auto& idx_0 = shape.mesh.indices[i * 3 + 0];
                const auto& idx_1 = shape.mesh.indices[i * 3 + 1];
                const auto& idx_2 = shape.mesh.indices[i * 3 + 2];

                vtx_info_list.push_back(idx_0);
                auto it = vtx_info_list.back();
                faceParam.v0.idx[0] = static_cast<uint32_t>(vtx_info_list.size()) - 1 + curr_vtx_pos;

                vtx_info_list.push_back(idx_1);
                it = vtx_info_list.back();
                faceParam.v0.idx[1] = static_cast<uint32_t>(vtx_info_list.size()) - 1 + curr_vtx_pos;

                vtx_info_list.push_back(idx_2);
                it = vtx_info_list.back();
                faceParam.v0.idx[2] = static_cast<uint32_t>(vtx_info_list.size()) - 1 + curr_vtx_pos;

                triangle_parameters.push_back(faceParam);
            }

            vec3 triangle_bbox_min = vec3(AT_MATH_INF);
            vec3 triangle_bbox_max = vec3(-AT_MATH_INF);

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
                    vtx.uv.z = need_compute_normal_on_the_fly ? float(1) : float(0);
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
                triangle_bbox_min = vec3(
                    std::min(triangle_bbox_min.x, vtx.pos.x),
                    std::min(triangle_bbox_min.y, vtx.pos.y),
                    std::min(triangle_bbox_min.z, vtx.pos.z));
                triangle_bbox_max = vec3(
                    std::max(triangle_bbox_max.x, vtx.pos.x),
                    std::max(triangle_bbox_max.y, vtx.pos.y),
                    std::max(triangle_bbox_max.z, vtx.pos.z));
            }

            // Compute bounding box of shape..
            shape_bbox_min = vec3(
                std::min(shape_bbox_min.x, triangle_bbox_min.x),
                std::min(shape_bbox_min.y, triangle_bbox_min.y),
                std::min(shape_bbox_min.z, triangle_bbox_min.z));
            shape_bbox_max = vec3(
                std::max(shape_bbox_max.x, triangle_bbox_max.x),
                std::max(shape_bbox_max.y, triangle_bbox_max.y),
                std::max(shape_bbox_max.z, triangle_bbox_max.z));

            std::shared_ptr<aten::TriangleGroupMesh> aten_shape;
            int32_t prev_mtrl_idx = -1;

            // One shape has one material.It means another shape would be created if different material appear.
            for (uint32_t i = 0; i < triangle_num; i++) {
                // Loading as triangle is specified, so vertex num per triangle have to be 3.
                AT_ASSERT(shape.mesh.num_face_vertices[i] == 3);

                int32_t mtrl_idx = shape.mesh.material_ids[i];

                if (mtrl_idx < 0 && !aten_shape) {
                    // If any material is not defined in obj file, try to create the material via the external function.
                    aten_shape = std::make_shared<aten::TriangleGroupMesh>();

                    std::shared_ptr<aten::material> regireterd_mtrl = nullptr;

                    if (callback_create_mtrl) {
                        regireterd_mtrl = callback_create_mtrl(
                            "", ctxt, MaterialType::Diffuse, aten::vec3(1), "", "");
                    }

                    AT_ASSERT(regireterd_mtrl);

                    if (regireterd_mtrl) {
                        aten_shape->SetMaterial(regireterd_mtrl);
                    }
                }
                else if (prev_mtrl_idx != mtrl_idx) {
                    // If the different material from the previous one appear.

                    if (aten_shape) {
                        // If the shape already exist.

                        auto mtrl = aten_shape->GetMaterial();

                        if (mtrl->param().type == aten::MaterialType::Emissive) {
                            // Export as the seprated object which has an emissive material.
                            auto emitobj(aten::TransformableFactory::createObject(ctxt));
                            emitobj->setName(shape.name);
                            emitobj->appendShape(aten_shape);
                            emitobj->setBoundingBox(aten::aabb(shape_bbox_min, shape_bbox_max));
                            objs.push_back(std::move(emitobj));
                        }
                        else {
                            // When different material from the previous one appear, register the shape to the object.
                            // And, create new shape and shift to it later.
                            create_obj_functor(obj, shape, ctxt);
                            obj->appendShape(aten_shape);
                        }
                    }

                    // Create a new shape for the next new material.
                    aten_shape = std::make_shared<aten::TriangleGroupMesh>();

                    if (mtrl_idx >= 0) {
                        // Apply new materil to the shape.
                        const auto& mtrl = mtrls[mtrl_idx];

                        auto aten_mtrl = callback_find_mtrl
                            ? callback_find_mtrl(mtrl.name, ctxt)
                            : ctxt.FindMaterialByName(mtrl.name);

                        // If the specifiend name material is not found, try to create a new material via the external function.
                        if (!aten_mtrl && callback_create_mtrl) {
                            std::shared_ptr<material> new_mtrl(
                                callback_create_mtrl(
                                    mtrl.name,
                                    ctxt,
                                    MaterialType::Diffuse,
                                    aten::vec3(mtrl.diffuse[0], mtrl.diffuse[1], mtrl.diffuse[2]),
                                    mtrl.diffuse_texname,
                                    mtrl.bump_texname));
                            aten_shape->SetMaterial(new_mtrl);
                        }
                        else {
                            aten_shape->SetMaterial(aten_mtrl);
                        }
                    }

                    // For the next loop.
                    prev_mtrl_idx = mtrl_idx;

                    // If the shape doesn't have a material until here, set dummy material....
                    if (!aten_shape->GetMaterial()) {
                        // Only diffuse.
                        const auto& objmtrl = mtrls[mtrl_idx];

                        std::shared_ptr<aten::material> mtrl;

                        if (callback_create_mtrl) {
                            mtrl = callback_create_mtrl(
                                    objmtrl.name,
                                    ctxt,
                                    MaterialType::Diffuse,
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
                                auto tex = ctxt.GetTextureByName(objmtrl.diffuse_texname);

                                if (tex) {
                                    albedoMap = tex.get();
                                }
                                else {
                                    std::string texname = pathname + "/" + objmtrl.diffuse_texname;
                                    auto loaded_img = aten::ImageLoader::load(texname, ctxt);
                                    albedoMap = loaded_img.get();
                                }
                            }

                            // Normal map.
                            if (!objmtrl.bump_texname.empty()) {
                                auto tex = ctxt.GetTextureByName(objmtrl.bump_texname);

                                if (tex) {
                                    normalMap = tex.get();
                                }
                                else {
                                    std::string texname = pathname + "/" + objmtrl.bump_texname;
                                    auto loaded_img = aten::ImageLoader::load(texname, ctxt);
                                    normalMap = loaded_img.get();
                                }
                            }

                            aten::MaterialParameter mtrlParam;
                            mtrlParam.type = aten::MaterialType::Diffuse;
                            mtrlParam.baseColor = diffuse;

                            mtrl = ctxt.CreateMaterialWithMaterialParameter(
                                objmtrl.name,
                                mtrlParam,
                                albedoMap,
                                normalMap,
                                nullptr);
                        }

                        aten_shape->SetMaterial(mtrl);
                    }
                }

                // Add a triangle to the shape(= triangle group which has the same material).
                auto& triangle_param = triangle_parameters[i];

                const auto& v0 = ctxt.GetVertex(triangle_param.v0.idx[0]);
                const auto& v1 = ctxt.GetVertex(triangle_param.v0.idx[1]);
                const auto& v2 = ctxt.GetVertex(triangle_param.v0.idx[2]);

                if (v0.uv.z == float(1)
                    || v1.uv.z == float(1)
                    || v2.uv.z == float(1)
                    || need_compute_normal_on_the_fly)
                {
                    triangle_param.v1.needNormal = 1;
                }

                // Triangle should belong the triangle group which has the same material(=shape).
                // So, apply the same material index.
                triangle_param.v1.mtrlid = aten_shape->GetMaterial()->id();
                triangle_param.v1.mesh_id = aten_shape->get_mesh_id();

                auto f = ctxt.CreateTriangle(triangle_param);

                aten_shape->AddFace(f);
            }

            // Register the shape to the object.
            {
                const auto& mtrl = aten_shape->GetMaterial();

                if (will_register_shape_as_separate_obj) {
                    // Register the each shape as the separated different obj.
                    create_obj_functor(obj, shape, ctxt);

                    obj->appendShape(aten_shape);
                    obj->setBoundingBox(aten::aabb(shape_bbox_min, shape_bbox_max));
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
                    emitobj->setName(shape.name);
                    emitobj->appendShape(aten_shape);
                    emitobj->setBoundingBox(aten::aabb(shape_bbox_min, shape_bbox_max));
                    objs.push_back(std::move(emitobj));
                }
                else {
                    create_obj_functor(obj, shape, ctxt);
                    obj->appendShape(aten_shape);
                }
            }
        }

        // Register all shapes as one obj.
        // In this case, obj instance has to be created.
        if (!will_register_shape_as_separate_obj && obj) {
            obj->setBoundingBox(aten::aabb(shape_bbox_min, shape_bbox_max));

            // TODO
            // If we need to manage if the same tagged object exist, should we register it here?

            objs.push_back(std::move(obj));
        }

        auto vtxNum = ctxt.GetVertexNum();

        AT_PRINTF("(%s)\n", path.data());
        AT_PRINTF("    %d[vertices]\n", vtxNum);
        AT_PRINTF("    %d[polygons]\n", all_triangle_num);

        return objs;
    }
}
