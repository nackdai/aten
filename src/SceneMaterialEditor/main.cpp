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

	aten::window::SetCurrentDirectoryFromExe();

	aten::AssetManager::suppressWarnings();

	MaterialEditWindow::init(
		1280, 720,
		"MaterialEdit");

	MaterialSelectWindow::init(
		1280, 720,
		"MaterialSelect",
		"../../asset/cornellbox/orig.obj");

	aten::window::run();

	aten::GLProfiler::terminate();

	aten::window::terminate();
}
