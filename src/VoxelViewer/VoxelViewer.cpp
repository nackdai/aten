#include "VoxelViewer.h"
#include "visualizer/atengl.h"

bool VoxelViewer::init(
    int32_t width, int32_t height,
    std::string_view pathVS,
    std::string_view pathFS)
{
    width_ = width;
    height_ = height;

    static const std::array attribs = {
        aten::VertexAttrib{ GL_FLOAT, 3, sizeof(GLfloat), 0 },
    };

    static const std::array VoxelVtxs = {
        aten::vec4(0,  1,  1, 1),
        aten::vec4(0,  0,  1, 1),
        aten::vec4(1,  1,  1, 1),
        aten::vec4(1,  0,  1, 1),
        aten::vec4(0,  1,  0, 1),
        aten::vec4(0,  0,  0, 1),
        aten::vec4(1,  1,  0, 1),
        aten::vec4(1,  0,  0, 1),
    };

    constexpr std::array VoxelIdxs = {
        // +X
        2, 3, 6,
        6, 3, 7,

        // -X
        0, 4, 1,
        4, 5, 1,

        // +Y
        4, 0, 6,
        6, 0, 2,

        // -Y
        1, 5, 3,
        3, 5, 7,

        // +Z
        0, 1, 2,
        2, 1, 3,

        // -Z
        6, 7, 4,
        4, 7, 5,
    };

    constexpr std::array VoxelWireFrameIdxs = {
        0, 1,
        0, 2,
        1, 3,
        2, 3,

        4, 5,
        4, 6,
        5, 7,
        6, 7,

        2, 6,
        3, 7,

        0, 4,
        1, 5,
    };

    // vertex buffer.
    vertex_buffer_.init(
        sizeof(decltype(VoxelVtxs)::value_type),
        VoxelVtxs.size(),
        0,
        attribs.data(),
        attribs.size(),
        VoxelVtxs.data());

    // index buffer.
    ib_.init(VoxelIdxs.size(), VoxelIdxs.data());

    ib_for_wireframe_.init(VoxelWireFrameIdxs.size(), VoxelWireFrameIdxs.data());

    return shader_.init(width, height, pathVS, pathFS);
}

void VoxelViewer::bringVoxels(
    const std::vector<aten::ThreadedSbvhNode>& nodes,
    std::vector<std::vector<aten::ThreadedSbvhNode>>& voxelList)
{
    for (const auto& node : nodes) {
        if (AT_IS_VOXEL(node.voxeldepth)) {
            int32_t depth = AT_GET_VOXEL_DEPTH(node.voxeldepth);
            voxelList[depth].push_back(node);
        }
    }
}

void VoxelViewer::draw(
    const aten::context& ctxt,
    const aten::camera* cam,
    std::vector<std::vector<aten::ThreadedSbvhNode>>& voxelList,
    bool isWireframe,
    uint32_t depth)
{
    shader_.prepareRender(nullptr, false);

    CALL_GL_API(::glEnable(GL_DEPTH_TEST));
    CALL_GL_API(::glEnable(GL_CULL_FACE));

    auto camparam = cam->param();

    // TODO
    camparam.znear = float(0.1);
    camparam.zfar = float(10000.0);

    aten::mat4 mtx_W2V;
    aten::mat4 mtx_V2C;

    mtx_W2V.lookat(
        camparam.origin,
        camparam.center,
        camparam.up);

    mtx_V2C.perspective(
        camparam.znear,
        camparam.zfar,
        camparam.vfov,
        camparam.aspect);

    aten::mat4 mtx_W2C = mtx_V2C * mtx_W2V;

    auto hMtxW2C = shader_.getHandle("mtx_W2C");
    CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, &mtx_W2C.a[0]));

    aten::mat4 mtx_L2W;

    auto hMtxL2W = shader_.getHandle("mtx_L2W");
    auto hColor = shader_.getHandle("color");
    auto hNormal = shader_.getHandle("normal");

    // NOTE
    // Box has 6 rectangles. Rectangle has 2 triangles.
    // Then, box has 12 (= 6 * 2) triangles (i.e. primitives).
    constexpr int32_t PrimCnt = 12;

    depth = (depth / aten::sbvh::VoxelDepth) * aten::sbvh::VoxelDepth;

    auto& voxels = voxelList[depth];

    for (size_t i = 0; i < voxels.size(); i++) {
        const auto& voxel = voxels[i];

        auto voxeldepth = (int32_t)AT_GET_VOXEL_DEPTH(voxel.voxeldepth);
        AT_ASSERT(voxeldepth == depth);

        if (voxeldepth == depth) {
            auto size = voxel.boxmax - voxel.boxmin;

            aten::mat4 mtxScale;
            aten::mat4 mtxTrans;

            mtxScale.asScale(size);
            mtxTrans.asTrans(voxel.boxmin);

            mtx_L2W = mtxTrans * mtxScale;

            aten::vec3 color(float(0));

            const auto mtrl = ctxt.GetMaterialInstance(voxel.mtrlid);

            if (mtrl) {
                color = mtrl->color();
            }

            CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, &mtx_L2W.a[0]));
            CALL_GL_API(::glUniform3f(hColor, color.x, color.y, color.z));

            // TODO
            CALL_GL_API(::glUniform3f(hNormal, 1.0f, 0.0f, 0.0f));

            if (isWireframe) {
                ib_for_wireframe_.draw(vertex_buffer_, aten::Primitive::Lines, 0, 12);
            }
            else {
                ib_.draw(vertex_buffer_, aten::Primitive::Triangles, 0, PrimCnt);
            }
        }
    }
}
