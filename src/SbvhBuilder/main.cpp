#include "aten.h"
#include "atenscene.h"

int main(int argc, char* argv[])
{
	aten::window::SetCurrentDirectoryFromExe();

	aten::AssetManager::suppressWarnings();

	auto obj = aten::ObjLoader::load("../../asset/suzanne/suzanne.obj");

	if (!obj) {
		// TODO
		return 0;
	}

	// Not use.
	// But specify internal accelerator type in constructor...
	aten::AcceleratedScene<aten::sbvh> scene;
	
	try {
		obj->exportInternalAccelTree("");
	}
	catch (std::exception* e) {
		// TODO

		return 0;
	}

	return 1;
}
