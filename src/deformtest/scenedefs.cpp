#include "scenedefs.h"
#include "atenscene.h"

static aten::instance<aten::deformable>* s_deformMdl = nullptr;

aten::instance<aten::deformable>* getDeformable()
{
	return s_deformMdl;
}

void ObjCornellBoxScene::makeScene(aten::scene* scene)
{
	auto emit = new aten::emissive(aten::vec3(36, 33, 24));
	aten::AssetManager::registerMtrl(
		"light",
		emit);

	aten::AssetManager::registerMtrl(
		"backWall",
		new aten::lambert(aten::vec3(0.580000, 0.568000, 0.544000)));
	aten::AssetManager::registerMtrl(
		"ceiling",
		new aten::lambert(aten::vec3(0.580000, 0.568000, 0.544000)));
	aten::AssetManager::registerMtrl(
		"floor",
		new aten::lambert(aten::vec3(0.580000, 0.568000, 0.544000)));
	aten::AssetManager::registerMtrl(
		"leftWall",
		new aten::lambert(aten::vec3(0.504000, 0.052000, 0.040000)));

	aten::AssetManager::registerMtrl(
		"rightWall",
		new aten::lambert(aten::vec3(0.112000, 0.360000, 0.072800)));
	aten::AssetManager::registerMtrl(
		"shortBox",
		new aten::lambert(aten::vec3(0.580000, 0.568000, 0.544000)));
	aten::AssetManager::registerMtrl(
		"tallBox",
		new aten::lambert(aten::vec3(0.580000, 0.568000, 0.544000)));

	std::vector<aten::object*> objs;

#if 1
	aten::ObjLoader::load(objs, "../../asset/cornellbox/orig.obj", true);

	auto light = new aten::instance<aten::object>(
		objs[0],
		aten::vec3(0),
		aten::vec3(0),
		aten::vec3(1));
	scene->add(light);

	auto areaLight = new aten::AreaLight(light, emit->param().baseColor);
	scene->addLight(areaLight);

	for (int i = 1; i < objs.size(); i++) {
		auto box = new aten::instance<aten::object>(objs[i], aten::mat4::Identity);
		scene->add(box);
	}
#else
	aten::ObjLoader::load(objs, "../../asset/cornellbox/orig_nolight.obj");
	auto box = new aten::instance<aten::object>(objs[0], aten::mat4::Identity);
	scene->add(box);
#endif
}

void ObjCornellBoxScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(0.f, 1.f, 3.f);
	at = aten::vec3(0.f, 1.f, 0.f);
	fov = 45;
}

/////////////////////////////////////////////////////

void DeformScene::makeScene(aten::scene* scene)
{
	aten::deformable* mdl = new aten::deformable();
	mdl->read("unitychan_gpu.mdl");

	aten::ImageLoader::setBasePath("../../asset/unitychan/Texture");
	aten::MaterialLoader::load("unitychan_mtrl.xml");

	auto deformMdl = new aten::instance<aten::deformable>(mdl, aten::mat4::Identity);
	scene->add(deformMdl);

	s_deformMdl = deformMdl;

	aten::ImageLoader::setBasePath("./");
}

void DeformScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(0, 71, 225);
	at = aten::vec3(0, 71, 216);
	fov = 45;
}
