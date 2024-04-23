#include "ObjWriter.h"
#include "utility.h"

//#pragma optimize( "", off)

namespace aten {
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
        int32_t pos{ -1 };
        int32_t nml{ -1 };
        int32_t uv{ -1 };

        ObjVertex() {}

        ObjVertex(int32_t p, int32_t n, int32_t u)
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
        // obj は 1 基点なので、+1 する.

        // NOTE
        // pos/uv/nml.

        fprintf(fp, "f ");

        for (int32_t i = 0; i < AT_COUNTOF(ObjFace::vtx); i++) {
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

    static inline void replaceIndex(ObjVertex& v, int32_t idx)
    {
        v.pos = v.pos >= 0 ? idx : -1;
        v.nml = v.nml >= 0 ? idx : -1;
        v.uv = v.uv >= 0 ? idx : -1;
    }

    bool ObjWriter::write(
        const std::string& path,
        const std::string& mtrlPath,
        const std::vector<aten::vertex>& vertices,
        const std::vector<std::vector<int32_t>>& indices,
        ObjWriter::FuncGetMtrlName func_get_mtrl_name)
    {
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

            auto mtrl_name = func_get_mtrl_name(i);

            // TODO
            // Write material name as group name.
            fprintf(fp, "g %s\n", mtrl_name);

            if (mtrl_name) {
                fprintf(fp, "usemtl %s\n", mtrl_name);
            }

            for (const auto& t : tris) {
                writeFace(fp, t);
            }

            writeLineFeed(fp);
        }

        fclose(fp);

        return true;
    }

    bool ObjWriter::writeObjects(
        const std::string& path,
        const std::string& mtrlPath,
        const context& ctxt,
        const std::vector<std::shared_ptr<aten::PolygonObject>>& objs)
    {
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
            writeLineFeed(fp);
        }

        uint32_t obj_idx = 0;
        uint32_t vtx_idx = 0;

        std::vector<ObjFace> export_face_vtx;

        for (const auto& obj : objs) {
            const auto& shapes = obj->getShapes();

            export_face_vtx.clear();

            size_t tri_num = 0;
            for (const auto& s : shapes) {
                const auto& tris = s->GetTriangleList();
                tri_num += tris.size();
            }
            export_face_vtx.reserve(tri_num);

            for (const auto& s : shapes) {
                const auto& tris = s->GetTriangleList();

                for (const auto& t : tris) {
                    const auto& tri_param = t->GetParam();

                    ObjFace triangle;

                    for (int32_t i = 0; i < 3; i++) {
                        const auto& v = ctxt.GetVertex(tri_param.idx[i]);

                        triangle.vtx[i].pos = writeVertexPosition(fp, v) ? vtx_idx : -1;
                        triangle.vtx[i].nml = writeVertexNormal(fp, v) ? vtx_idx : -1;
                        triangle.vtx[i].uv = writeVertexUV(fp, v) ? vtx_idx : -1;

                        vtx_idx++;
                    }

                    export_face_vtx.push_back(triangle);
                }
            }

            writeLineFeed(fp);

            const auto obj_name = obj->getName();
            if (!obj_name.empty()) {
                fprintf(fp, "g %s\n", obj_name);
            }
            else {
                fprintf(fp, "g %d\n", obj_idx++);
            }

            size_t tri_pos = 0;

            for (const auto& s : shapes) {
                auto mtrl_name = s->GetMaterial()->name();
                fprintf(fp, "usemtl %s\n", mtrl_name);

                const auto& tris = s->GetTriangleList();

                for (size_t i = 0; i < tris.size(); i++) {
                    const auto& triangle = export_face_vtx[tri_pos];
                    writeFace(fp, triangle);

                    tri_pos++;
                }
            }

            writeLineFeed(fp);
        }

        fclose(fp);

        return true;
    }

    bool ObjWriter::runOnThread(
        std::function<void()> funcFinish,
        const std::string& path,
        const std::string& mtrlPath,
        const std::vector<aten::vertex>& vertices,
        const std::vector<std::vector<int32_t>>& indices,
        FuncGetMtrlName func_get_mtrl_name)
    {
        if (m_isRunning) {
            // Not finish yet.
            return false;
        }

        static std::vector<std::vector<int32_t>> tmpIdx;

        if (!m_thread.isRunning()) {
            m_thread.start([&](void* data) {
                while (1) {
                    m_sema.wait();

                    if (m_isTerminate) {
                        break;
                    }

                    write(
                        path,
                        mtrlPath,
                        vertices,
                        indices,
                        func_get_mtrl_name);

                    if (funcFinish) {
                        funcFinish();
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
    }
}
