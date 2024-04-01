#pragma once

#include "aten.h"

class VoxelViewer {
public:
    VoxelViewer() = default;
    ~VoxelViewer() = default;

public:
    // 初期化.
    bool init(
        int32_t width, int32_t height,
        std::string_view pathVS,
        std::string_view pathFS);

    // SBVHのノードからボクセルデータを持つもののみを取り出す.
    // 対象となるのは、BottomLayerの１つのみ.
    void bringVoxels(
        const std::vector<aten::ThreadedSbvhNode>& nodes,
        std::vector<std::vector<aten::ThreadedSbvhNode>>& voxelList);

    // 取り出されたボクセルデータについて描画する.
    void draw(
        const aten::context& ctxt,
        const aten::camera* cam,
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
