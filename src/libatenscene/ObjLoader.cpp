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

        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> mtrls;
        std::string err;

        // TODO
        // mtl_basepath

        //auto flags = tinyobj::triangulation | tinyobj::calculate_normals;
        auto flags = tinyobj::triangulation;

        auto result = tinyobj::LoadObj(
            shapes, mtrls,
            err,
            path.c_str(), mtrlBasePath.c_str(),
            flags);

        //AT_VRETURN(result, );
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

            auto vtxnum = shape.mesh.positions.size();

            auto curVtxPos = ctxt.getVertexNum();

            vec3 pmin = vec3(AT_MATH_INF);
            vec3 pmax = vec3(-AT_MATH_INF);

            // positions and normals.
            for (uint32_t i = 0; i < vtxnum; i += 3) {
                vertex vtx;

                vtx.pos.x = shape.mesh.positions[i + 0];
                vtx.pos.y = shape.mesh.positions[i + 1];
                vtx.pos.z = shape.mesh.positions[i + 2];
                vtx.pos.w = real(0);

                if (shape.mesh.normals.empty()) {
                    // Flag not to specify normal.
                    vtx.uv.z = real(1);
                }
                else {
                    vtx.nml.x = shape.mesh.normals[i + 0];
                    vtx.nml.y = shape.mesh.normals[i + 1];
                    vtx.nml.z = shape.mesh.normals[i + 2];
                    vtx.uv.z = needComputeNormalOntime ? real(1) : real(0);
                }

                if (std::isnan(vtx.nml.x) || std::isnan(vtx.nml.y) || std::isnan(vtx.nml.z))
                {
                    // TODO
                    // work around...
                    vtx.nml = aten::vec4(real(0), real(1), real(0), 1);
                }

                ctxt.addVertex(vtx);

                pmin = vec3(
                    std::min(pmin.x, vtx.pos.x),
                    std::min(pmin.y, vtx.pos.y),
                    std::min(pmin.z, vtx.pos.z));
                pmax = vec3(
                    std::max(pmax.x, vtx.pos.x),
                    std::max(pmax.y, vtx.pos.y),
                    std::max(pmax.z, vtx.pos.z));
            }

            shapemin = vec3(
                std::min(shapemin.x, pmin.x),
                std::min(shapemin.y, pmin.y),
                std::min(shapemin.z, pmin.z));
            shapemax = vec3(
                std::max(shapemax.x, pmax.x),
                std::max(shapemax.y, pmax.y),
                std::max(shapemax.z, pmax.z));

            aten::objshape* dstshape = nullptr; 
            int mtrlidx = -1;

            auto idxnum = shape.mesh.indices.size();

            for (uint32_t i = 0; i < idxnum; i += 3) {
                int mtrlpos = i / 3;

                int m = shape.mesh.material_ids[mtrlpos];

                if (mtrlidx != m) {
                    if (dstshape) {
                        // Check if emmisive object.
                        auto mtrl = dstshape->getMaterial();

                        if (willSeparate) {
                            obj->appendShape(dstshape);
                            obj->setBoundingBox(aten::aabb(pmin, pmax));
                            objs.push_back(obj);

                            obj = aten::TransformableFactory::createObject(ctxt);
                        }
                        if (mtrl->param().type == aten::MaterialType::Emissive) {
                            auto emitobj = aten::TransformableFactory::createObject(ctxt);
                            emitobj->appendShape(dstshape);
                            emitobj->setBoundingBox(aten::aabb(pmin, pmax));
                            objs.push_back(emitobj);
                        }
                        else {
                            obj->appendShape(dstshape);
                        }
                    }

                    dstshape = new aten::objshape();
                    mtrlidx = m;

                    if (mtrlidx >= 0) {
                        const auto mtrl = mtrls[mtrlidx];
                        dstshape->setMaterial(AssetManager::getMtrl(mtrl.name));
                    }

                    if (!dstshape->getMaterial()) {
                        // No material, set dummy material....

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

                        dstshape->setMaterial(mtrl);

                        AssetManager::registerMtrl(mtrl->name(), mtrl);
                    }
                }

                aten::PrimitiveParamter faceParam;

                faceParam.idx[0] = shape.mesh.indices[i + 0] + curVtxPos;
                faceParam.idx[1] = shape.mesh.indices[i + 1] + curVtxPos;
                faceParam.idx[2] = shape.mesh.indices[i + 2] + curVtxPos;

                auto& v0 = ctxt.getVertex(faceParam.idx[0]);
                auto& v1 = ctxt.getVertex(faceParam.idx[1]);
                auto& v2 = ctxt.getVertex(faceParam.idx[2]);

                if (v0.uv.z == real(1)
                    || v1.uv.z == real(1)
                    || v2.uv.z == real(1))
                {
                    faceParam.needNormal = 1;
                }

                faceParam.mtrlid = dstshape->getMaterial()->id();
                faceParam.gemoid = dstshape->getGeomId();

                auto f = ctxt.createTriangle(faceParam);

                dstshape->addFace(f);
            }

            // Keep polygon counts.
            numPolygons += idxnum / 3;

            {
                auto mtrl = dstshape->getMaterial();

                if (willSeparate) {
                    obj->appendShape(dstshape);
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
                    auto emitobj = aten::TransformableFactory::createObject(ctxt);
                    emitobj->appendShape(dstshape);
                    emitobj->setBoundingBox(aten::aabb(pmin, pmax));
                    objs.push_back(emitobj);
                }
                else {
                    obj->appendShape(dstshape);
                }
            }

            // texture cooridnates.
            vtxnum = shape.mesh.texcoords.size();

            if (vtxnum > 0) {
                for (uint32_t i = 0; i < vtxnum; i += 2) {
                    uint32_t vpos = i / 2 + curVtxPos;

                    auto& vtx = ctxt.getVertex(vpos);

                    vtx.uv.x = shape.mesh.texcoords[i + 0];
                    vtx.uv.y = shape.mesh.texcoords[i + 1];
                }
            }
            else {
                // NOTE
                // positions ‚É‚Í x,y,z ‚ª‚Î‚ç‚Î‚ç‚É“ü‚Á‚Ä‚¢‚é‚Ì‚ÅA3 ‚ÅŠ„‚é‚±‚Æ‚Å‚P’¸“_‚ ‚½‚è‚É‚È‚é.
                vtxnum = shape.mesh.positions.size() / 3;

                for (uint32_t i = 0; i < vtxnum; i++) {
                    uint32_t vpos = i + curVtxPos;

                    auto& vtx = ctxt.getVertex(vpos);

                    // Specify not have texture coordinates.
                    vtx.uv.z = real(-1);
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
