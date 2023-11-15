#include <cmdline.h>

#include "aten.h"
#include "atenscene.h"
#include "ObjWriter.h"

struct Options {
    std::string input;
    std::string output;
};

bool parseOption(
    int32_t argc, char* argv[],
    cmdline::parser& cmd,
    Options& opt)
{
    {
        cmd.add<std::string>("input", 'i', "input filename", true);
        cmd.add<std::string>("output", 'o', "output filename base", false, "result");

        cmd.add<std::string>("help", '?', "print usage", false);
    }

    bool isCmdOk = cmd.parse(argc, argv);

    if (cmd.exist("help")) {
        std::cerr << cmd.usage();
        return false;
    }

    if (!isCmdOk) {
        std::cerr << cmd.error() << std::endl << cmd.usage();
        return false;
    }

    if (cmd.exist("input")) {
        opt.input = cmd.get<std::string>("input");
    }
    else {
        std::cerr << cmd.error() << std::endl << cmd.usage();
        return false;
    }

    if (cmd.exist("output")) {
        opt.output = cmd.get<std::string>("output");
    }
    else {
        // TODO
        opt.output = "result.sbvh";
    }

    return true;
}

int32_t main(int32_t argc, char* argv[])
{
    Options opt;
    cmdline::parser cmd;

#if 0
    if (!parseOption(argc, argv, cmd, opt)) {
        return 0;
    }
#endif

    aten::SetCurrentDirectoryFromExe();

    aten::AssetManager::suppressWarnings();

    aten::context ctxt;

    std::vector<std::shared_ptr<aten::PolygonObject>> objs;
    aten::ObjLoader::load(
        objs,
        "../../asset/mansion/interior_bundled4_chairmove_1163769_606486_2.obj",
        ctxt);

    if (objs.empty()) {
        // TODO
        return 0;
    }

    // TODO
    static const char* names[] = {
        "Screen",
        "base",
        "Keyboard",
        "Middle",
        "Silver",
    };

    auto& obj = objs[0];

    std::vector<aten::TriangleGroupMesh*> shapes;
    std::vector<std::shared_ptr<aten::material>> mtrls;

    const auto& meshes = obj->getShapes();

    for (const auto mesh : meshes) {
        auto mtrl = mesh->GetMaterial();

        std::string mtrlName(mtrl->name());

        for (auto name : names) {
            if (mtrlName == name) {
                shapes.push_back(mesh.get());
                mtrls.push_back(mtrl);

                break;
            }
        }
    }

    struct IndexMapper {
        int32_t orgIdx;
        int32_t newIdx;

        IndexMapper(int32_t o, int32_t n) : orgIdx(o), newIdx(n) {}
    };

    std::vector<IndexMapper> indexMapList;

    // Gather vertices.
    std::vector<aten::vertex> vertices;
    {
        // Gather indices for gathering vertices.
        std::vector<int32_t> tmpIndices(shapes.size());

        for (auto shape : shapes) {
            const auto& tris = shape->GetTriangleList();

            for (auto tri : tris) {
                const auto& triParam = tri->GetParam();

                for (int32_t i = 0; i < 3; i++) {
                    auto idx = triParam.idx[i];
                    auto found = std::find(tmpIndices.begin(), tmpIndices.end(), idx);
                    if (found == tmpIndices.end()) {
                        tmpIndices.push_back(idx);
                    }
                }
            }
        }

        indexMapList.reserve(tmpIndices.size());

        // Gather vertices.
        for (auto idx : tmpIndices) {
            const auto v = ctxt.GetVertex(idx);

            int32_t newIdx = vertices.size();
            vertices.push_back(v);

            // Keep index map.
            indexMapList.push_back(IndexMapper(idx, newIdx));
        }
    }

    // Gather indices.
    std::vector<std::vector<int32_t>> indices(shapes.size());
    {
        for (int32_t s = 0; s < shapes.size(); s++) {
            auto shape = shapes[s];
            const auto& tris = shape->GetTriangleList();

            for (auto tri : tris) {
                const auto& triParam = tri->GetParam();

                for (int32_t i = 0; i < 3; i++) {
                    auto idx = triParam.idx[i];

                    auto found = std::find_if(
                        indexMapList.begin(),
                        indexMapList.end(),
                        [&](const IndexMapper& idxMapper)
                    {
                        return idxMapper.orgIdx == idx;
                    });

                    AT_ASSERT(found != indexMapList.end());

                    int32_t newIdx = found->newIdx;
                    indices[s].push_back(newIdx);
                }
            }
        }
    }

    // NOTE
    // obj ファイル向けにはマテリアルはshapeと同じオーダーでリストに格納されていないといけない.
    // つまり、重複しても構わない.
    // しかし、mtl ファイル向けには重複しているのはよろしくない.

    // 重複を許さないマテリアルリストの作成.
    std::vector<aten::material*> mtrlListForExport;
    for (auto& mtrl : mtrls) {
        if (std::find(mtrlListForExport.begin(), mtrlListForExport.end(), mtrl.get()) == mtrlListForExport.end()) {
            mtrlListForExport.push_back(mtrl.get());
        }
    }

    // Export to obj.
    aten::ObjWriter::write(
        "result.obj",
        "result.mtl",
        vertices,
        indices,
        [&mtrls](uint32_t idx) {
            return mtrls[idx]->name();
        });

    aten::ObjWriter::writeMaterial(
        ctxt,
        "result.mtl",
        mtrlListForExport);

    return 1;
}
