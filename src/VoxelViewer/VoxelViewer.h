#pragma once

#include "aten.h"

class VoxelViewer {
public:
    VoxelViewer() = default;
    ~VoxelViewer() = default;

public:
    // ������.
    bool init(
        int32_t width, int32_t height,
        std::string_view pathVS,
        std::string_view pathFS);

    // SBVH�̃m�[�h����{�N�Z���f�[�^�������݂̂̂����o��.
    // �ΏۂƂȂ�̂́ABottomLayer�̂P�̂�.
    void bringVoxels(
        const std::vector<aten::ThreadedSbvhNode>& nodes,
        std::vector<std::vector<aten::ThreadedSbvhNode>>& voxelList);

    // ���o���ꂽ�{�N�Z���f�[�^�ɂ��ĕ`�悷��.
    void draw(
        const aten::context& ctxt,
        const aten::Camera* cam,
        std::vector<std::vector<aten::ThreadedSbvhNode>>& voxelList,
        bool isWireframe,
        uint32_t depth);

private:
    aten::shader shader_;
    aten::GeomVertexBuffer vertex_buffer_;
    aten::GeomIndexBuffer ib_;

    aten::GeomIndexBuffer ib_for_wireframe_;

    int32_t width_{ 0 };
    int32_t height_{ 0 };
};
