#include <vector>
#include "aten.h"
#include "atenscene.h"

int main(int argc, char* argv[])
{
	aten::window::SetCurrentDirectoryFromExe();

	aten::ImageLoader::setBasePath("../../asset/");

	auto mtrl = aten::MaterialLoader::load("material.json");
	AT_ASSERT(mtrl);

	auto mtrl2 = aten::AssetManager::getMtrl("material");
	AT_ASSERT(mtrl == mtrl2);
}
