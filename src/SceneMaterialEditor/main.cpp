#include "MaterialEditWindow.h"
#include "MaterialSelectWindow.h"

#include "atenscene.h"

#ifdef ENABLE_OMP
static uint32_t g_threadnum = 8;
#else
static uint32_t g_threadnum = 1;
#endif

int main()
{
    aten::timer::init();
    aten::OMPUtil::setThreadNum(g_threadnum);

    aten::SetCurrentDirectoryFromExe();

    aten::AssetManager::suppressWarnings();

#if 1
    MaterialEditWindow::init(
        1280, 720,
        "MaterialEdit");

    aten::material::clearMaterialList();
#endif

    MaterialSelectWindow::init(
        1280, 720,
        "MaterialSelect",
        //"../../asset/cornellbox/orig.obj");
        //"../../asset/sponza/sponza.obj");
        "../../asset/mansion/interior_bundled4_chairmove_1163769_606486_2.obj",
        "./material.xml");
        //"../../asset/mansion/objs/room.obj");

    MaterialEditWindow::buildScene();

    aten::material::clearMaterialList();

    MaterialSelectWindow::setFuncPickMtrlIdNotifier(
        MaterialEditWindow::notifyPickMtrlId);

    MaterialEditWindow::setFuncChangeMtrlIdNotifier(
        MaterialSelectWindow::notifyChangeMtrlId);

    aten::window::run();

    aten::GLProfiler::terminate();

    aten::window::terminate();
}
