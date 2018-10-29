#include "ObjWriter.h"
#include "utility.h"

//#pragma optimize( "", off)

static inline void writeLineFeed(FILE* fp)
{
    fprintf(fp, "\n");
}

static inline bool writeVertexPosition(FILE* fp, const aten::vertex& vtx)
{
    fprintf(fp, "v  %f %f %f\n", vtx.pos.x, vtx.pos.y, vtx.pos.z);
    return true;
}

static inline bool writeVertexNormal(FILE* fp, const aten::vertex& vtx)
{
    auto l = aten::squared_length(vtx.nml);

    if (l > 0) {
        fprintf(fp, "vn  %f %f %f\n", vtx.nml.x, vtx.nml.y, vtx.nml.z);
        return true;
    }

    return false;
}

static inline bool writeVertexUV(FILE* fp, const aten::vertex& vtx)
{
    bool hasTexCoord = vtx.uv.z >= 0;

    if (hasTexCoord) {
        fprintf(fp, "vt  %f %f 0.000000\n", vtx.uv.x, vtx.uv.y);
        return true;
    }

    return false;
}

struct ObjVertex {
    int pos{ -1 };
    int nml{ -1 };
    int uv{ -1 };

    ObjVertex() {}

    ObjVertex(int p, int n, int u)
    {
        pos = p;
        nml = n;
        uv = u;
    }
};

struct ObjFace {
    ObjVertex vtx[3];

    ObjFace() {}
};

static inline void writeFace(FILE* fp, const ObjFace& f)
{
    // NOTE
    // obj ÇÕ 1 äÓì_Ç»ÇÃÇ≈ÅA+1 Ç∑ÇÈ.

    // NOTE
    // pos/uv/nml.

    fprintf(fp, "f ");

    for (int i = 0; i < AT_COUNTOF(ObjFace::vtx); i++) {
        fprintf(fp, "%d/", f.vtx[i].pos + 1);

        if (f.vtx[i].uv >= 0) {
            fprintf(fp, "%d", f.vtx[i].uv + 1);
        }

        fprintf(fp, "/");

        if (f.vtx[i].nml >= 0) {
            fprintf(fp, "%d", f.vtx[i].nml + 1);
        }

        fprintf(fp, " ");
    }

    writeLineFeed(fp);
}

static inline void writeMaterrialTag(FILE* fp, const aten::material* mtrl)
{
    auto name = mtrl->name();
    fprintf(fp, "usemtl %s\n", name);
}

static inline void replaceIndex(ObjVertex& v, int idx)
{
    v.pos = v.pos >= 0 ? idx : -1;
    v.nml = v.nml >= 0 ? idx : -1;
    v.uv = v.uv >= 0 ? idx : -1;
}

bool ObjWriter::write(
    const std::string& path,
    const std::string& mtrlPath,
    const std::vector<aten::vertex>& vertices,
    const std::vector<std::vector<int>>& indices,
    const std::vector<aten::material*>& mtrls)
{
    AT_ASSERT(mtrls.size() == indices.size());

    std::string mtrlPathName;
    std::string extname;
    std::string filename;

    aten::getStringsFromPath(
        mtrlPath,
        mtrlPathName,
        extname,
        filename);

    FILE* fp = fopen(path.c_str(), "wt");

    // Write header.
    {
        if (mtrlPathName == "./") {
            mtrlPathName = mtrlPath;
        }

        fprintf(fp, "mtllib %s\n", mtrlPathName.c_str());
    }

    std::vector<ObjVertex> vtxs;

    // Write Vertices.
    for (uint32_t i = 0; i < vertices.size(); i++) {
        const auto& v = vertices[i];

        bool hasPos = writeVertexPosition(fp, v);
        bool hasNml = writeVertexNormal(fp, v);
        bool hasUv = writeVertexUV(fp, v);

        // Set tentative index.
        vtxs.push_back(
            ObjVertex(
                hasPos ? i : -1,
                hasNml ? i : -1,
                hasUv ? i : -1));
    }

    std::vector<std::vector<ObjFace>> triGroup(indices.size());

    // Make faces.
    for (uint32_t i = 0; i < triGroup.size(); i++) {
        auto& tris = triGroup[i];
        const auto& idxs = indices[i];

        tris.reserve(idxs.size());

        for (uint32_t n = 0; n < idxs.size(); n += 3) {
            auto id0 = idxs[n + 0];
            auto id1 = idxs[n + 1];
            auto id2 = idxs[n + 2];

            ObjFace t;
            t.vtx[0] = vtxs[id0];
            t.vtx[1] = vtxs[id1];
            t.vtx[2] = vtxs[id2];

            // Replace correct index.
            replaceIndex(t.vtx[0], id0);
            replaceIndex(t.vtx[1], id1);
            replaceIndex(t.vtx[2], id2);

            tris.push_back(t);
        }
    }

    // Write faces.
    for (uint32_t i = 0; i < triGroup.size(); i++) {
        const auto& tris = triGroup[i];

        const auto mtrl = mtrls[i];

        // TODO
        // Write dummy group name...
        fprintf(fp, "g %d\n", i);

        writeMaterrialTag(fp, mtrl);

        for (const auto& t : tris) {
            writeFace(fp, t);
        }

        writeLineFeed(fp);
    }

    fclose(fp);

    return true;
}

bool ObjWriter::writeMaterial(
    const aten::context& ctxt,
    const std::string& mtrlPath,
    const std::vector<aten::material*>& mtrls)
{
    FILE* fp = fopen(mtrlPath.c_str(), "wt");

    for (const auto* mtrl : mtrls) {
        fprintf(fp, "newmtl %s\n", mtrl->name());

        const auto& param = mtrl->param();

        fprintf(fp, "Ns 1.000000\n");
        fprintf(fp, "Ka 0.000000 0.000000 0.000000\n");
        fprintf(fp, "Kd %.6f %.6f %.6f\n", param.baseColor.x, param.baseColor.y, param.baseColor.z);
        fprintf(fp, "Ks 0.000000 0.000000 0.000000\n");
        fprintf(fp, "Ni 1.000000\n");
        fprintf(fp, "d 1.000000\n");
        fprintf(fp, "illum 2\n");

        if (param.albedoMap >= 0) {
            auto albedo = ctxt.getTexture(param.albedoMap);
            fprintf(fp, "map_Ka %s\n", albedo->name());
            fprintf(fp, "map_Kd %s\n", albedo->name());
        }

        if (param.normalMap >= 0) {
            auto normal = ctxt.getTexture(param.normalMap);
            fprintf(fp, "map_bump %s\n", normal->name());
            fprintf(fp, "bump %s\n", normal->name());
        }

        fprintf(fp, "\n");
    }

    fclose(fp);

    return true;
}

bool ObjWriter::runOnThread(
    std::function<void()> funcFinish,
    const std::string& path,
    const std::string& mtrlPath,
    const std::vector<aten::vertex>& vertices,
    const std::vector<std::vector<int>>& indices,
    const std::vector<aten::material*>& mtrls)
{
    if (m_isRunning) {
        // Not finish yet.
        return false;
    }

    if (m_param) {
        delete m_param;
        m_param = nullptr;
    }
    m_param = new WriteParams(funcFinish, path, mtrlPath, vertices, indices, mtrls);

    static std::vector<std::vector<int>> tmpIdx;

    if (!m_thread.isRunning()) {
        m_thread.start([&](void* data) {
            while (1) {
                m_sema.wait();

                if (m_isTerminate) {
                    break;
                }

                write(
                    m_param->path,
                    m_param->mtrlPath,
                    m_param->vertices,
                    m_param->indices,
                    m_param->mtrls);

                if (m_param->funcFinish) {
                    m_param->funcFinish();
                }

                m_isRunning = false;
            }

        }, nullptr);
    }

    m_isRunning = true;
    m_sema.notify();

    return true;
}

void ObjWriter::terminate()
{
    m_isTerminate = true;

    // Awake thread, maybe thread does not run.
    m_sema.notify();

    m_thread.join();

    delete m_param;
    m_param = nullptr;
}
