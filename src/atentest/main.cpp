#include <vector>
#include "aten.h"
#include "atenscene.h"

int main(int argc, char* argv[])
{
	aten::window::SetCurrentDirectoryFromExe();

	aten::ImageLoader::setBasePath("../../asset/");

	//aten::MaterialLoader::load("material.json");
	aten::MaterialLoader::load("material.xml");

	auto mtrl0 = aten::AssetManager::getMtrl("test");
	AT_ASSERT(mtrl0);

	auto mtrl1 = aten::AssetManager::getMtrl("test2");
	AT_ASSERT(mtrl1);

	auto mtrl2 = aten::AssetManager::getMtrl("test3");
	AT_ASSERT(mtrl2);

	auto info = aten::SceneLoader::load("scene.json");
}
