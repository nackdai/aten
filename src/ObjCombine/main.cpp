#include <cmdline.h>

#include "aten.h"
#include "atenscene.h"
#include "ObjWriter.h"

struct Args {
    std::string input;
    std::string combine;
    std::string output;
    std::string mtrl;

    Args() = default;
    Args(std::string_view i, std::string_view c, std::string_view o, std::string_view m)
        : input(i), combine(c), output(o), mtrl(m)
    {}
};

std::optional<Args> ParseArgs(int32_t argc, char* argv[])
{
    Args args;

    cmdline::parser cmd;
    {
        cmd.add<std::string>("input", 'i', "input filename", true);
        cmd.add<std::string>("combine", 'c', "combine filename", true);
        cmd.add<std::string>("output", 'o', "output filename", false, "result");

        cmd.add<std::string>("help", '?', "print usage", false);
    }

    if (cmd.exist("help")) {
        std::cerr << cmd.usage();
        return std::nullopt;
    }

    const auto is_suceeded = cmd.parse(argc, argv);

    if (!is_suceeded) {
        std::cerr << cmd.error() << std::endl << cmd.usage();
        return std::nullopt;
    }

    args.input = cmd.get<std::string>("input");
    args.combine = cmd.get<std::string>("combine");

    if (cmd.exist("output")) {
        args.output = cmd.get<std::string>("output");
    }
    else {
        args.output = "result.obj";
    }

    std::string pathname;
    std::string extname;
    std::string filename;

    aten::getStringsFromPath(
        args.output,
        pathname, extname, filename);
    args.mtrl = aten::StringFormat("sc.mtl", pathname);

    return args;
}

int32_t main(int32_t argc, char* argv[])
{
#if 1
    auto args = ParseArgs(argc, argv);

    if (!args.has_value()) {
        AT_ASSERT(false);
        return 1;
    }
#else
    auto args = std::make_shared<Args>(
        "../../asset/cornellbox/box.obj",
        "../../asset/cornellbox/bunny.obj",
        "bunny_in_box.obj",
        "bunny_in_box.mtl"
    );
#endif

    aten::SetCurrentDirectoryFromExe();

    aten::context ctxt;

    auto objs = aten::ObjLoader::Load(args->input, ctxt);

    if (objs.empty()) {
        // TODO
        AT_ASSERT(false);
        return 1;
    }

    auto combine_objs = aten::ObjLoader::Load(args->combine, ctxt);

    if (combine_objs.empty()) {
        // TODO
        AT_ASSERT(false);
        return 1;
    }

    // TODO:
    // Overlap the material name.

    std::vector<aten::TriangleGroupMesh*> shapes;
    std::vector<std::shared_ptr<aten::material>> mtrls;

    auto func_obtain_shapes_mtrls = [&shapes, &mtrls](const decltype(objs)& obj_list) {
        for (const auto& obj : obj_list) {
            auto meshes = obj->getShapes();
            for (const auto mesh : meshes) {
                auto mtrl = mesh->GetMaterial();

                shapes.push_back(mesh.get());
                mtrls.push_back(mtrl);
            }
        }
    };

    func_obtain_shapes_mtrls(objs);
    func_obtain_shapes_mtrls(combine_objs);

    // Gather indices.
    std::vector<std::vector<int32_t>> indices(shapes.size());
    for (size_t shape_idx = 0; shape_idx < shapes.size(); shape_idx++) {
        const auto& shape = shapes[shape_idx];
        const auto& tris = shape->GetTriangleList();

        for (size_t tri_idx = 0; tri_idx < tris.size(); tri_idx++) {
            const auto& tri = tris[tri_idx];
            const auto& tri_param = tri->GetParam();

            indices[shape_idx].push_back(tri_param.idx[0]);
            indices[shape_idx].push_back(tri_param.idx[1]);
            indices[shape_idx].push_back(tri_param.idx[2]);
        }
    }

    const auto& vertices = ctxt.GetVertices();

    // Export to obj.
    aten::ObjWriter::write(
        args->output,
        args->mtrl,
        vertices,
        indices,
        [&mtrls](uint32_t idx) {
            return mtrls[idx]->name();
        });

    aten::ObjWriter::writeMaterial(
        ctxt,
        args->mtrl,
        mtrls);

    return 0;
}
