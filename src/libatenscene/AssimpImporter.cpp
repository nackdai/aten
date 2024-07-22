#include <memory>

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"

#include "AssimpImporter.h"

namespace aten
{
    bool AssimpImporter::load(
        std::string_view path,
        std::vector<std::shared_ptr<aten::PolygonObject>>& objs,
        context& ctxt,
        FuncCreateMaterial func_create_mtrl)
    {
        AssimpImporter importer;
        return importer.loadModel(path, objs, ctxt, func_create_mtrl);
    }

    std::string getTextureName(aiTextureType texture_type, const aiMaterial* assimp_mtrl) {
        auto tex_count = assimp_mtrl->GetTextureCount(texture_type);

        if (tex_count > 0) {
            if (tex_count != 1) {
                AT_ASSERT(false);
                AT_PRINTF("Allow one texture per type\n");
            }
            else {
                aiString assimp_string;
                assimp_mtrl->GetTexture(texture_type, 0, &assimp_string);
                std::string str(assimp_string.data);

                return str;
            }
        }

        return std::string();
    }

    const std::string CreateMaterial(
        context& ctxt,
        const aiMaterial* assimp_mtrl,
        AssimpImporter::FuncCreateMaterial func_create_mtrl)
    {
        aiString name;
        assimp_mtrl->Get(AI_MATKEY_NAME, name);

        std::string mtrl_name(name.C_Str());
        AT_PRINTF("%s\n", mtrl_name.c_str());

        auto stored_mtrl = ctxt.FindMaterialByName(mtrl_name);
        if (stored_mtrl) {
            // The specified material already exists
            return std::string(stored_mtrl->name());
        }

        MaterialParameter mtrl_param;
        mtrl_param.type = MaterialType::Lambert;

        bool is_emissive = false;

        aiColor3D emissive;
        if (assimp_mtrl->Get(AI_MATKEY_COLOR_EMISSIVE, emissive) == AI_SUCCESS) {
            mtrl_param.baseColor = vec3(emissive.r, emissive.g, emissive.b);

            // If emmsive color is invalid, the material is treated as emissive.
            is_emissive = dot((vec3)mtrl_param.baseColor, (vec3)mtrl_param.baseColor) > float(0);

            if (is_emissive) {
                mtrl_param.type = MaterialType::Emissive;
            }
        }

        if (!is_emissive){
            // TODO
            // Should I use disney material?

            aiColor3D albedo;
            if (assimp_mtrl->Get(AI_MATKEY_COLOR_DIFFUSE, albedo) == AI_SUCCESS) {
                mtrl_param.type = MaterialType::Lambert;
                mtrl_param.baseColor = vec3(albedo.r, albedo.g, albedo.b);
            }

            float opacity = 0.0f;;
            if (assimp_mtrl->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS) {
                mtrl_param.baseColor.a = opacity;
            }

            float ior = 0.0f;
            if (assimp_mtrl->Get(AI_MATKEY_REFRACTI, ior) == AI_SUCCESS) {
                mtrl_param.type = MaterialType::Refraction;
                mtrl_param.standard.ior = ior;
            }

            float shininess = 0.0f;
            if (assimp_mtrl->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS) {
                AT_ASSERT(false);
                mtrl_param.type = MaterialType::Beckman;
                mtrl_param.standard.roughness = shininess;
            }
        }

        auto albedo_tex_name(std::move(getTextureName(aiTextureType::aiTextureType_DIFFUSE, assimp_mtrl)));
        auto normal_tex_name(std::move(getTextureName(aiTextureType::aiTextureType_NORMALS, assimp_mtrl)));

        auto mtrl = func_create_mtrl(mtrl_name, ctxt, mtrl_param, albedo_tex_name, normal_tex_name);
        if (mtrl) {
            return mtrl_name;
        }

        return std::string();
    }

    bool createObject(
        context& ctxt,
        std::vector<std::shared_ptr<aten::PolygonObject>>& objs,
        const aiNode* assimp_node,
        const aiScene* assimp_scene,
        const std::vector<std::string>& mtrl_list)
    {
        if (assimp_node->mNumMeshes > 0) {
            // Make transform matrix with traversing to the root parent.
            auto transform_mtx = assimp_node->mTransformation;
            const auto* parent = assimp_node->mParent;
            while (parent) {
                transform_mtx *= parent->mTransformation;
                parent = parent->mParent;
            }

            aten::mat4 mtx(
                transform_mtx.a1, transform_mtx.a2, transform_mtx.a3, transform_mtx.a4,
                transform_mtx.b1, transform_mtx.b2, transform_mtx.b3, transform_mtx.b4,
                transform_mtx.c1, transform_mtx.c2, transform_mtx.c3, transform_mtx.c4,
                transform_mtx.d1, transform_mtx.d2, transform_mtx.d3, transform_mtx.d4
            );

            auto obj = aten::TransformableFactory::createObject(ctxt);
            auto shape = std::make_shared<aten::TriangleGroupMesh>();

            aten::vec3 obj_min(AT_MATH_INF);
            aten::vec3 obj_max(-AT_MATH_INF);

            for (uint32_t i = 0; i < assimp_node->mNumMeshes; i++) {
                // Get mesh id which node has.
                uint32_t mesh_id = assimp_node->mMeshes[i];

                // Actual mesh is stored in the scene.
                const auto assimp_mesh = assimp_scene->mMeshes[mesh_id];

                // TODO
                if (assimp_mesh->HasBones()) {
                    AT_ASSERT(false);
                    AT_PRINTF("Not allow bone model\n");
                    return false;
                }

                if (!assimp_mesh->HasPositions()) {
                    AT_ASSERT(false);
                    AT_PRINTF("No position in mesh verticesl\n");
                    return false;
                }

                auto num_uv_channels = assimp_mesh->GetNumUVChannels();

                if (num_uv_channels > 0 && num_uv_channels != 1) {
                    AT_ASSERT(false);
                    AT_PRINTF("Invalid UV in mesh verticesl\n");
                    return false;
                }

                const auto vtx_idx_offset = ctxt.GetVertexNum();

                aten::vec3 aabb_min(AT_MATH_INF);
                aten::vec3 aabb_max(-AT_MATH_INF);

                for (uint32_t v_idx = 0; v_idx < assimp_mesh->mNumVertices; v_idx++) {
                    vertex vtx;

                    const auto& pos = assimp_mesh->mVertices[v_idx];
                    vtx.pos = aten::vec4(pos.x, pos.y, pos.z, float(1));
                    vtx.pos = mtx.apply(vtx.pos);
                    vtx.pos.w = float(0);

                    if (assimp_mesh->HasNormals()) {
                        const auto& nml = assimp_mesh->mNormals[v_idx];
                        vtx.nml = aten::vec3(nml.x, nml.y, nml.z);
                        vtx.nml = mtx.applyXYZ(vtx.nml);
                        vtx.nml = normalize(vtx.nml);
                    }
                    else {
                        // Flag not to specify normal.
                        vtx.uv.z = float(1);
                    }

                    if (assimp_mesh->HasTextureCoords(0)) {
                        vtx.uv.x = assimp_mesh->mTextureCoords[0][v_idx].x;
                        vtx.uv.y = assimp_mesh->mTextureCoords[0][v_idx].y;
                    }
                    else {
                        // Specify not have texture coordinates.
                        vtx.uv.z = float(-1);
                    }

                    ctxt.AddVertex(vtx);

                    // To compute bouding box, store min/max position.
                    aabb_min = vec3(
                        std::min(aabb_min.x, vtx.pos.x),
                        std::min(aabb_min.y, vtx.pos.y),
                        std::min(aabb_min.z, vtx.pos.z));
                    aabb_max = vec3(
                        std::max(aabb_max.x, vtx.pos.x),
                        std::max(aabb_max.y, vtx.pos.y),
                        std::max(aabb_max.z, vtx.pos.z));
                }

                AT_ASSERT(assimp_mesh->mMaterialIndex < mtrl_list.size());
                const auto mtrl_name = mtrl_list[assimp_mesh->mMaterialIndex];

                auto mtrl = ctxt.FindMaterialByName(mtrl_name);
                if (!mtrl) {
                    AT_ASSERT(false);
                    AT_PRINTF("Not found material [%s]\n", mtrl_name.c_str());
                    return false;
                };

                shape->SetMaterial(mtrl);

                for (uint32_t f_idx = 0; f_idx < assimp_mesh->mNumFaces; f_idx++) {
                    const auto& assimp_face = assimp_mesh->mFaces[f_idx];

                    if (assimp_face.mNumIndices != 3) {
                        AT_ASSERT(false);
                        AT_PRINTF("Face has to be triangle\n");
                        return false;
                    }

                    aten::TriangleParameter face_param;

                    face_param.idx[0] = static_cast<uint32_t>(assimp_mesh->mFaces[f_idx].mIndices[0]) + vtx_idx_offset;
                    face_param.idx[1] = static_cast<uint32_t>(assimp_mesh->mFaces[f_idx].mIndices[1]) + vtx_idx_offset;
                    face_param.idx[2] = static_cast<uint32_t>(assimp_mesh->mFaces[f_idx].mIndices[2]) + vtx_idx_offset;

                    const auto& v0 = ctxt.GetVertex(face_param.idx[0]);
                    const auto& v1 = ctxt.GetVertex(face_param.idx[1]);
                    const auto& v2 = ctxt.GetVertex(face_param.idx[2]);

                    if (v0.uv.z == float(1)
                        || v1.uv.z == float(1)
                        || v2.uv.z == float(1))
                    {
                        face_param.needNormal = 1;
                    }

                    face_param.mtrlid = shape->GetMaterial()->id();
                    face_param.mesh_id = shape->get_mesh_id();

                    auto f = ctxt.CreateTriangle(face_param);

                    shape->AddFace(f);
                }

                if (mtrl->param().type == aten::MaterialType::Emissive) {
                    // Export the object which has an emissive material as the emissive object.
                    auto emit_obj = aten::TransformableFactory::createObject(ctxt);
                    emit_obj->appendShape(shape);
                    emit_obj->setBoundingBox(aten::aabb(aabb_min, aabb_max));
                    objs.push_back(std::move(emit_obj));
                }
                else {
                    obj->appendShape(shape);
                }
            }

            obj->setBoundingBox(aten::aabb(obj_min, obj_max));
            objs.push_back(std::move(obj));
        }

        for (uint32_t child_idx = 0; child_idx < assimp_node->mNumChildren; child_idx++) {
            const auto child = assimp_node->mChildren[child_idx];
            if (!createObject(ctxt, objs, assimp_node, assimp_scene, mtrl_list)) {
                return false;
            }
        }

        return true;
    }

    bool AssimpImporter::loadModel(
        std::string_view path,
        std::vector<std::shared_ptr<aten::PolygonObject>>& objs,
        context& ctxt,
        FuncCreateMaterial func_create_mtrl)
    {
        uint32_t assimp_flags = aiProcessPreset_TargetRealtime_MaxQuality |
            aiProcess_OptimizeGraph |
            aiProcess_FlipUVs;

        // Never use Assimp's tangent gen code
        assimp_flags &= ~(aiProcess_CalcTangentSpace);

        Assimp::Importer importer;
        auto assimp_scene = importer.ReadFile(path.data(), assimp_flags);

        auto is_valid_scene = [](const decltype(assimp_scene) scene) {
            if (scene->mTextures) {
                AT_PRINTF("Not allow internal textures\n");
                return false;
            }
            return true;
        };

        if (!assimp_scene || !is_valid_scene(assimp_scene)) {
            AT_PRINTF("Can't open model file [%s]\n", path.data());
            AT_PRINTF("  %s\n", importer.GetErrorString());
            return false;
        }

        for (uint32_t i = 0; i < assimp_scene->mNumMaterials; i++) {
            const auto assimp_mtrl = assimp_scene->mMaterials[i];
            auto mtrl_name = CreateMaterial(ctxt, assimp_mtrl, func_create_mtrl);
            if (!mtrl_name.empty()) {
                mtrl_list_.push_back(std::move(mtrl_name));
            }
        }

        auto root_node = assimp_scene->mRootNode;
        if (!createObject(ctxt, objs, root_node, assimp_scene, mtrl_list_)) {
            return false;
        }

        return true;
    }
}
