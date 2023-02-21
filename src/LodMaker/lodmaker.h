#pragma once

#include <vector>
#include "aten.h"

class LodMaker {
public:
    LodMaker() {}
    ~LodMaker() {}

public:
    static void make(
        std::vector<aten::vertex>& dstVertices,
        std::vector<std::vector<int32_t>>& dstIndices,
        const aten::aabb& bound,
        const std::vector<aten::vertex>& vertices,
        const std::vector<std::vector<aten::face*>>& tris,
        int32_t gridX,
        int32_t gridY,
        int32_t gridZ);

    static void removeCollapsedTriangles(
        std::vector<std::vector<int32_t>>& dstIndices,
        const std::vector<aten::vertex>& vertices,
        const std::vector<std::vector<int32_t>>& indices);

    bool runOnThread(
        std::function<void()> funcFinish,
        std::vector<aten::vertex>& dstVertices,
        std::vector<std::vector<int32_t>>& dstIndices,
        const aten::aabb& bound,
        const std::vector<aten::vertex>& vertices,
        const std::vector<std::vector<aten::face*>>& tris,
        int32_t gridX,
        int32_t gridY,
        int32_t gridZ);

    void terminate();

    bool isRunningThread() const
    {
        return m_isRunning;
    }

private:
    struct LodParams {
        std::function<void()> funcFinish;

        std::vector<aten::vertex>& dstVertices;
        std::vector<std::vector<int32_t>>& dstIndices;
        aten::aabb bound;
        const std::vector<aten::vertex>& vertices;
        const std::vector<std::vector<aten::face*>>& tris;
        int32_t gridX;
        int32_t gridY;
        int32_t gridZ;

        LodParams(
            std::function<void()> func,
            std::vector<aten::vertex>& _dstVertices,
            std::vector<std::vector<int32_t>>& _dstIndices,
            const aten::aabb& _bound,
            const std::vector<aten::vertex>& _vertices,
            const std::vector<std::vector<aten::face*>>& _tris,
            int32_t x, int32_t y, int32_t z)
            : dstVertices(_dstVertices),
            dstIndices(_dstIndices),
            vertices(_vertices),
            tris(_tris)
        {
            funcFinish = func;
            bound = _bound;
            gridX = x;
            gridY = y;
            gridZ = z;
        }
    };

    LodParams* m_param{ nullptr };

    aten::Thread m_thread;
    aten::Semaphore m_sema;

    std::atomic<bool> m_isRunning{ false };
    std::atomic<bool> m_isTerminate{ false };
};
