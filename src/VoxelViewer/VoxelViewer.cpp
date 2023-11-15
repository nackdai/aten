#include "VoxelViewer.h"
#include "visualizer/atengl.h"

static const aten::vec4 VoxelVtxs[] = {
    aten::vec4( 0,  1,  1, 1),
    aten::vec4( 0,  0,  1, 1),
    aten::vec4( 1,  1,  1, 1),
    aten::vec4( 1,  0,  1, 1),

    aten::vec4(0,  1,  0, 1),
    aten::vec4(0,  0,  0, 1),
    aten::vec4(1,  1,  0, 1),
    aten::vec4(1,  0,  0, 1),
};

static const uint32_t VoxelIdxs[] = {
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

static const uint32_t VoxelWireFrameIdxs[] = {
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


bool VoxelViewer::init(
    int32_t width, int32_t height,
    std::string_view pathVS,
    std::string_view pathFS)
{
    width_ = width;
    height_ = height;

    static const aten::VertexAttrib attribs[] = {
        { GL_FLOAT, 3, sizeof(GLfloat), 0 },
    };

    // vertex buffer.
    vertex_buffer_.init(
        sizeof(aten::vec4),
        AT_COUNTOF(VoxelVtxs),
        0,
        attribs,
        AT_COUNTOF(attribs),
        VoxelVtxs);

    // index buffer.
    m_ib.init(AT_COUNTOF(VoxelIdxs), VoxelIdxs);

    m_ibForWireframe.init(AT_COUNTOF(VoxelWireFrameIdxs), VoxelWireFrameIdxs);

    return m_shader.init(width, height, pathVS, pathFS);
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
    m_shader.prepareRender(nullptr, false);

    CALL_GL_API(::glEnable(GL_DEPTH_TEST));
    CALL_GL_API(::glEnable(GL_CULL_FACE));

    auto camparam = cam->param();

    // TODO
    camparam.znear = real(0.1);
    camparam.zfar = real(10000.0);

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

    auto hMtxW2C = m_shader.getHandle("mtx_W2C");
    CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, &mtx_W2C.a[0]));

    aten::mat4 mtx_L2W;

    auto hMtxL2W = m_shader.getHandle("mtx_L2W");
    auto hColor = m_shader.getHandle("color");
    auto hNormal = m_shader.getHandle("normal");

    // NOTE
    // Box�Ȃ̂ŁA�P�ʓ�����O�p�`�Q�ŁA�U��.
    static const int32_t PrimCnt = 2 * 6;

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

            aten::vec3 color(real(0));

            const auto mtrl = ctxt.GetMaterialInstance(voxel.mtrlid);

            if (mtrl) {
                color = mtrl->color();
            }

            CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, &mtx_L2W.a[0]));
            CALL_GL_API(::glUniform3f(hColor, color.x, color.y, color.z));

            // TODO
            CALL_GL_API(::glUniform3f(hNormal, 1.0f, 0.0f, 0.0f));

            if (isWireframe) {
                m_ibForWireframe.draw(vertex_buffer_, aten::Primitive::Lines, 0, 12);
            }
            else {
                m_ib.draw(vertex_buffer_, aten::Primitive::Triangles, 0, PrimCnt);
            }
        }
    }
}
