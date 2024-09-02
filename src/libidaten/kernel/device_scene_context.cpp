#include "kernel/device_scene_context.cuh"

namespace idaten {
    void DeviceContextInHost::BindToDeviceContext()
    {
        if (!ctxt.shapes) {
            ctxt.lightnum = static_cast<int32_t>(lightparam.num());

            std::vector<cudaTextureObject_t> tmp_node;
            for (auto& node : nodeparam) {
                auto nodeTex = node.bind();
                tmp_node.push_back(nodeTex);
            }
            nodetex.writeFromHostToDeviceByNum(tmp_node.data(), tmp_node.size());

            ctxt.nodes = nodetex.data();

            if (!texRsc.empty())
            {
                std::vector<cudaTextureObject_t> tmp_tex;
                for (auto& rsc : texRsc) {
                    auto cudaTex = rsc.bind();
                    tmp_tex.push_back(cudaTex);
                }
                tex.writeFromHostToDeviceByNum(tmp_tex.data(), tmp_tex.size());
            }
            ctxt.textures = tex.data();
        }

        if (!grids.empty() && !ctxt.grid_holder.IsGridsAssigned()) {
            ctxt.grid_holder.AssignGrids(grids.data(), grids.num());
        }

        ctxt.vtxPos = vtxparamsPos.bind();
        ctxt.vtxNml = vtxparamsNml.bind();

        for (auto& node : nodeparam) {
            std::ignore = node.bind();
        }
    }
}
