#include "scenedefs.h"
#include "atenscene.h"

void DemoScene::makeScene(
    aten::context& ctxt,
    aten::scene* scene)
{
    aten::ImageLoader::setBasePath("../../asset/mansion/objs");

    aten::MaterialLoader::load("../../asset/mansion/objs/mansion_mtrl.xml", ctxt);

    static struct ObjInfo {
        const char* name;
        bool needComputeNormalOntime;
    } objinfo[] = {
        { "bed", false },
        { "chair", false },
        { "column_0", true },
        { "column_1", true },
        { "desk", true },
        { "lamp_board", true },
        { "lamp_body", false },
        { "mirror", true },
        { "pc", false },
        { "room", true }, 
        { "side_table", true },
        { "sofa_0", true },
        { "pillow_0", false },
        { "sofa_1", true },
        { "pillow_1", false },
        { "vase_on_desk", false },
        { "vase_on_sidetable", false },
        { "window_0", true },
        { "window_1", true },
    };

    aten::ObjLoader::setBasePath("../../asset/mansion/objs");

    for (auto info : objinfo) {
        std::vector<aten::object*> objs;

        std::string objpath(info.name);
        objpath += ".obj";

        std::string sbvhpath(info.name);
        sbvhpath = "../../asset/mansion/objs/" + sbvhpath + ".sbvh";

        aten::ObjLoader::load(objs, objpath.c_str(), ctxt, false, info.needComputeNormalOntime);
        objs[0]->importInternalAccelTree(sbvhpath.c_str(), ctxt, 0);

        auto inst = aten::TransformableFactory::createInstance<aten::object>(ctxt, objs[0], aten::mat4::Identity);

        scene->add(inst);
    }

    aten::ImageLoader::setBasePath("./");
}

void DemoScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    real& fov)
{
    //pos = aten::vec3(0.f, 1.f, 3.f);
    //at = aten::vec3(0.f, 1.f, 0.f);
    pos = aten::vec3(-2.47f,- 0.11f, 0.876f);
    at = aten::vec3(-1.49f, -0.32f, 0.837f);
    fov = 45;
}
