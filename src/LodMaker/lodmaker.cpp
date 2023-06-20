#include "lodmaker.h"

struct QVertex {
    aten::vertex v;
    uint32_t idx;       // vertex index.
    uint32_t orgIdx;    // for debug. original vertex index.
    uint32_t grid[3];   // grid index.
    int32_t group;          // triangle group.
    uint64_t hash;

    QVertex() {}

    QVertex(
        aten::vertex _v,
        uint32_t i,
        uint32_t original,
        int32_t g,
        uint64_t h,
        uint32_t gx, uint32_t gy, uint32_t gz)
        : v(_v), idx(i), hash(h)
    {
        orgIdx = original;
        group = g;
        grid[0] = gx;
        grid[1] = gy;
        grid[2] = gz;
    }

    const QVertex& operator=(const QVertex& rhs)
    {
        v = rhs.v;
        idx = rhs.idx;
        orgIdx = rhs.orgIdx;
        group = rhs.group;
        grid[0] = rhs.grid[0];
        grid[1] = rhs.grid[1];
        grid[2] = rhs.grid[2];
        hash = rhs.hash;

        return *this;
    }
};

// NOTE
// gridX * gridY * gridZ �� uint64_t �𒴂��Ȃ��悤�ɂ���

using qit = std::vector<QVertex>::iterator;

// Replaces the position of the vertex within the specified range
// with the position of the vertex closest to the average of the vertices within that range.
static void replaceVtxPositionToClosestFromAvg(qit start, qit end)
{
    aten::vec4 avg(start->v.pos);

    uint32_t cnt = 1;

    for (auto q = start + 1; q != end; q++) {
        avg += q->v.pos;

        cnt++;
    }

    avg /= cnt;

    real distMin = AT_MATH_INF;

    qit closest = start;

    // Find out the closest vertex to the average.
    for (auto q = start; q != end; q++) {
        auto d = (avg - q->v.pos).length();
        if (d < distMin) {
            distMin = d;
            closest = q;
        }
    }

    // Replace to the closest vertex position.
    for (auto q = start; q != end; q++) {
        q->v = closest->v;
    }
}

void LodMaker::make(
    std::vector<aten::vertex>& dstVertices,
    std::vector<std::vector<int32_t>>& dstIndices,
    const aten::aabb& bound,
    const std::vector<aten::vertex>& vertices,
    const std::vector<std::vector<aten::face*>>& triGroups,
    int32_t gridX,
    int32_t gridY,
    int32_t gridZ)
{
    auto bmin = bound.minPos();
    auto range = bound.size();

    aten::vec3 scale(
        (gridX - 1) / range.x,
        (gridY - 1) / range.y,
        (gridZ - 1) / range.z);

    static std::vector<QVertex> qvtxs;
    static std::vector<uint32_t> sortedIndices;

    qvtxs.clear();
    sortedIndices.clear();

    uint32_t orderIdx = 0;

    // �O���[�v�ɕ�����Ă���f�[�^����`���X�g�Ɋi�[����.

    for (uint32_t i = 0; i < triGroups.size(); i++) {
        const auto tris = triGroups[i];

        for (uint32_t n = 0; n < tris.size(); n++) {
            const auto tri = tris[n];

            const auto& triparam = tri->getParam();

            for (int32_t t = 0; t < 3; t++) {
                uint32_t vtxIdx = triparam.idx[t];
                const auto& v = vertices[vtxIdx];

                auto grid = ((aten::vec3)v.pos - bmin) * scale + real(0.5);

                uint32_t gx = (uint32_t)grid.x;
                uint32_t gy = (uint32_t)grid.y;
                uint32_t gz = (uint32_t)grid.z;

                uint64_t hash = gz * (gridX * gridY) + gy * gridX + gx;

                qvtxs.push_back(QVertex(v, orderIdx, vtxIdx, i, hash, gx, gy, gz));

                sortedIndices.push_back(orderIdx);
                orderIdx++;
            }
        }
    }

    // �����O���b�h�ɓ����Ă��钸�_�̏��ɂȂ�悤�Ƀ\�[�g����.

    {
        std::sort(
            qvtxs.begin(), qvtxs.end(),
            [](const QVertex& q0, const QVertex& q1)
        {
            if (q0.hash == q1.hash) {
                return q0.group > q1.group;
            }
            else {
                return q0.hash > q1.hash;
            }
        });

        uint32_t num = (uint32_t)qvtxs.size();

        // �C���f�b�N�X�����_�ɂ��킹�ĕ��בւ���.
        for (uint32_t n = 0; n < num; n++) {
            const auto& q = qvtxs[n];

            // ���X�̃C���f�b�N�X�̈ʒu�ɐV�����C���f�b�N�X�l������.
            sortedIndices[q.idx] = n;
        }
    }

    // �����O���b�h���ɓ����Ă��钸�_�̕��ϒl���v�Z���āA�P�̒��_�ɂ��Ă��܂�.
    // UV�͕���邪�A���F�Z�J���_���o�E���X�Ɏg���̂ŁA�����͋C�ɂ��Ȃ�.
    {
        auto start = qvtxs.begin();

        while (start != qvtxs.end())
        {
            auto end = start;

            // �O���b�h���قȂ钸�_�ɂȂ�܂ŒT��.
            while (end != qvtxs.end() && start->hash == end->hash) {
                end++;
            }

            // Replaces the position of the vertex within the specified range
            // with the position of the vertex closest to the average of the vertices within that range.
            replaceVtxPositionToClosestFromAvg(start, end);

            // Find out if there is something not replaced.
            for (auto q = start; q != end; q++) {
                if (q == start) {
                    dstVertices.push_back(q->v);
                }
                else {
                    // �قȂ�ʒu�̓_�̏ꍇ.

                    // �O�̓_�Ɣ�r����.
                    auto prev = q - 1;

                    auto v0 = q->v.pos;
                    auto v1 = prev->v.pos;

                    bool isEqual = (memcmp(&v0, &v1, sizeof(v0)) == 0);

                    if (!isEqual) {
                        dstVertices.push_back(q->v);
                    }
                }

                // �C���f�b�N�X�X�V.
                q->idx = (uint32_t)dstVertices.size() - 1;
            }

            // ���̊J�n�ʒu���X�V
            start = end;
        }
    }

    // LOD���ꂽ���ʂ̃C���f�b�N�X���i�[.

    dstIndices.resize(triGroups.size());

    orderIdx = 0;
    int32_t prevGroup = -1;

    // ���`�ɂȂ�񂾃C���f�b�N�X���O���[�v���ƂɐU�蕪����.

    for (uint32_t i = 0; i < triGroups.size(); i++) {
        const auto tris = triGroups[i];

        dstIndices[i].reserve(tris.size() * 3);

        for (uint32_t n = 0; n < tris.size(); n++) {
            const auto tri = tris[n];

            for (int32_t t = 0; t < 3; t++) {
                auto sortedIdx = sortedIndices[orderIdx];

                const auto& qvtx = qvtxs[sortedIdx];

                if (prevGroup < 0) {
                    prevGroup = qvtx.group;
                }

                auto newIdx = qvtx.idx;

                if (prevGroup == qvtx.group) {
                    dstIndices[qvtx.group].push_back(newIdx);
                }
                else {
                    // �O���[�v���ς����.

                    auto num = dstIndices[prevGroup].size();

                    // �O�p�`�ɂ��邽�߂ɕs�����Ă���_�̐�.
                    auto rest = 3 - num % 3;

                    if (rest == 1) {
                        dstIndices[prevGroup].push_back(newIdx);
                    }
                    else if (rest == 2) {
                        dstIndices[prevGroup].push_back(newIdx);

                        // �����K�v.
                        auto _sortedIdx = sortedIndices[orderIdx + 1];
                        auto _newIdx = qvtxs[sortedIdx].idx;

                        dstIndices[prevGroup].push_back(_newIdx);
                    }

                    dstIndices[qvtx.group].push_back(newIdx);
                }

                orderIdx++;

                prevGroup = qvtx.group;
            }
        }
    }
}

void LodMaker::removeCollapsedTriangles(
    std::vector<std::vector<int32_t>>& dstIndices,
    const std::vector<aten::vertex>& vertices,
    const std::vector<std::vector<int32_t>>& indices)
{
    dstIndices.resize(indices.size());

    for (int32_t i = 0; i < indices.size(); i++) {
        const auto& idxs = indices[i];

        for (int32_t n = 0; n < idxs.size(); n += 3) {
            auto id0 = idxs[n + 0];
            auto id1 = idxs[n + 1];
            auto id2 = idxs[n + 2];

            const auto& v0 = vertices[id0];
            const auto& v1 = vertices[id1];
            const auto& v2 = vertices[id2];

            // �O�p�`�̖ʐ� = �Q�ӂ̊O�ς̒��� / 2;
            auto e0 = v1.pos - v0.pos;
            auto e1 = v2.pos - v0.pos;
            auto area = real(0.5) * cross(e0, e1).length();

            if (area > real(0)) {
                dstIndices[i].push_back(id0);
                dstIndices[i].push_back(id1);
                dstIndices[i].push_back(id2);
            }
        }
    }
}

bool LodMaker::runOnThread(
    std::function<void()> funcFinish,
    std::vector<aten::vertex>& dstVertices,
    std::vector<std::vector<int32_t>>& dstIndices,
    const aten::aabb& bound,
    const std::vector<aten::vertex>& vertices,
    const std::vector<std::vector<aten::face*>>& tris,
    int32_t gridX,
    int32_t gridY,
    int32_t gridZ)
{
    if (m_isRunning) {
        // Not finish yet.
        return false;
    }

    if (m_param) {
        delete m_param;
        m_param = nullptr;
    }
    m_param = new LodParams(funcFinish, dstVertices, dstIndices, bound, vertices, tris, gridX, gridY, gridZ);

    static std::vector<std::vector<int32_t>> tmpIdx;

    if (!m_thread.isRunning()) {
        m_thread.start([&](void* data) {
            while (1) {
                m_sema.wait();

                if (m_isTerminate) {
                    break;
                }

                tmpIdx.clear();

                make(
                    m_param->dstVertices,
                    tmpIdx,
                    m_param->bound,
                    m_param->vertices,
                    m_param->tris,
                    m_param->gridX,
                    m_param->gridY,
                    m_param->gridZ);

                removeCollapsedTriangles(
                    m_param->dstIndices,
                    m_param->dstVertices,
                    tmpIdx);

                if (m_param->funcFinish) {
                    m_param->funcFinish();
                }

                m_isRunning = false;
            }

        }, nullptr);
    }

    dstVertices.clear();

    for (auto ids : dstIndices) {
        ids.clear();
    }
    dstIndices.clear();

    m_isRunning = true;
    m_sema.notify();

    return true;
}

void LodMaker::terminate()
{
    m_isTerminate = true;

    // Awake thread, maybe thread does not run.
    m_sema.notify();

    m_thread.join();

    delete m_param;
    m_param = nullptr;
}
