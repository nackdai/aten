#include "scenedefs.h"

static aten::instance<aten::object>* g_movableObj = nullptr;

aten::instance<aten::object>* getMovableObj()
{
	return g_movableObj;
}

void CornellBoxScene::makeScene(aten::scene* scene)
{
	auto emit = new aten::emissive(aten::vec3(36, 36, 36));
	//auto emit = new aten::emissive(aten::vec3(3, 3, 3));

	auto light = new aten::sphere(
		aten::vec3(50.0, 75.0, 81.6),
		5.0,
		emit);

	double r = 1e3;

	auto left = new aten::sphere(
		aten::vec3(r + 1, 40.8, 81.6),
		r,
		new aten::lambert(aten::vec3(0.75f, 0.25f, 0.25f)));

	auto right = new aten::sphere(
		aten::vec3(-r + 99, 40.8, 81.6),
		r,
		new aten::lambert(aten::vec3(0.25, 0.25, 0.75)));

	auto wall = new aten::sphere(
		aten::vec3(50, 40.8, r),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

	auto floor = new aten::sphere(
		aten::vec3(50, r, 81.6),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

	auto ceil = new aten::sphere(
		aten::vec3(50, -r + 81.6, 81.6),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

	//auto tex = aten::ImageLoader::load("../../asset/earth.bmp");

#define DEFALT	(1)

#if DEFALT
	// óŒãÖ.
	auto green = new aten::sphere(
		aten::vec3(65, 20, 20),
		20,
		new aten::lambert(aten::vec3(0.25, 0.75, 0.25)));
		//new aten::lambert(aten::vec3(1, 1, 1), tex));
#else
	auto green = new aten::sphere(
		aten::vec3(65, 20, 20),
		20,
		new aten::MicrofacetGGX(aten::vec3(0.7, 0.6, 0.5), 0.2, 0.2));
#endif

#if DEFALT
	// ãæ.
	auto mirror = new aten::sphere(
		aten::vec3(27, 16.5, 47),
		16.5,
		new aten::specular(aten::vec3(0.99, 0.99, 0.99)));
#else
	auto spec = new aten::MicrofacetBlinn(aten::vec3(1, 1, 1), 200, 0.8);
	auto diff = new aten::lambert(aten::vec3(0.0, 0.7, 0.0));

	auto layer = new aten::LayeredBSDF();
	layer->add(spec);
	layer->add(diff);

	auto mirror = new aten::sphere(
		aten::vec3(27, 16.5, 47),
		16.5,
		layer);
#endif

//#if DEFALT
#if 0
	// ÉKÉâÉX.
	auto glass = new aten::sphere(
		aten::vec3(77, 16.5, 78),
		16.5,
		new aten::refraction(aten::vec3(0.99, 0.99, 0.99), 1.5));
#elif 0
	auto glass = new aten::sphere(
		aten::vec3(77, 16.5, 78),
		5,
		emit);
#else
	aten::AssetManager::registerMtrl(
		"m1",
		//new aten::MicrofacetBlinn(aten::vec3(0.7, 0.6, 0.5), 200, 0.2));
		new aten::lambert(aten::vec3(0.2, 0.2, 0.7)));

	aten::AssetManager::registerMtrl(
		"Material.001",
		//new aten::MicrofacetBlinn(aten::vec3(0.7, 0.6, 0.5), 200, 0.2));
		new aten::lambert(aten::vec3(0.2, 0.2, 0.7)));

	auto obj = aten::ObjLoader::load("../../asset/suzanne/suzanne.obj");
	//auto obj = aten::ObjLoader::load("../../asset/teapot.obj");

	aten::mat4 mtxL2W;
	mtxL2W.asRotateByY(Deg2Rad(-25));

	aten::mat4 mtxT;
	mtxT.asTrans(aten::vec3(77, 16.5, 78));

	aten::mat4 mtxS;
	mtxS.asScale(10);

	mtxL2W = mtxT * mtxL2W * mtxS;

	auto glass = new aten::instance<aten::object>(obj, mtxL2W);
#endif

	scene->add(light);
#if 1
	scene->add(left);
	scene->add(right);
	scene->add(wall);
	scene->add(floor);
	scene->add(ceil);
	scene->add(green);
	scene->add(mirror);
#endif
	scene->add(glass);

#if 1
	aten::Light* l = new aten::AreaLight(light, emit->color());
#else
	aten::Light* l = new aten::AreaLight(glass, emit->color());
#endif

	scene->addLight(l);
}

void CornellBoxScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(50.0, 52.0, 295.6);
	at = aten::vec3(50.0, 40.8, 119.0);
	fov = 30;
}

/////////////////////////////////////////////////////

void RandomScene::makeScene(aten::scene* scene)
{
	auto s = new aten::sphere(aten::vec3(0, -1000, 0), 1000, new aten::lambert(aten::vec3(0.8, 0.8, 0.8)));
	scene->add(s);

	int i = 1;
	for (int x = -11; x < 11; x++) {
		for (int z = -11; z < 11; z++) {
			auto choose_mtrl = aten::drand48();

			aten::vec3 center = aten::vec3(
				x + 0.9 * aten::drand48(),
				0.2,
				z + 0.9 * aten::drand48());

			if (length(center - aten::vec3(4, 0.2, 0)) > 0.9) {
				if (choose_mtrl < 0.8) {
					// lambert
					s = new aten::sphere(
						center,
						0.2,
						new aten::lambert(aten::vec3(aten::drand48() * aten::drand48(), aten::drand48() * aten::drand48(), aten::drand48() * aten::drand48())));
				}
				else if (choose_mtrl < 0.95) {
					// specular
					s = new aten::sphere(
						center,
						0.2,
						new aten::specular(aten::vec3(0.5 * (1 + aten::drand48()), 0.5 * (1 + aten::drand48()), 0.5 * (1 + aten::drand48()))));
				}
				else {
					// glass
					s = new aten::sphere(center, 0.2, new aten::refraction(aten::vec3(1), 1.5));
				}

				scene->add(s);
			}
		}
	}

	s = new aten::sphere(aten::vec3(0, 1, 0), 1.0, new aten::refraction(aten::vec3(1), 1.5));
	scene->add(s);

	s = new aten::sphere(aten::vec3(-4, 1, 0), 1.0, new aten::lambert(aten::vec3(0.4, 0.2, 0.1)));
	scene->add(s);

	s = new aten::sphere(aten::vec3(4, 1, 0), 1.0, new aten::specular(aten::vec3(0.7, 0.6, 0.5)));
	scene->add(s);
}

void RandomScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(13, 2, 3);
	at = aten::vec3(0, 0, 0);
	fov = 30;
}

/////////////////////////////////////////////////////

void MtrlTestScene::makeScene(aten::scene* scene)
{
	auto s_blinn = new aten::sphere(aten::vec3(-1, 0, 0), 1.0, new aten::MicrofacetBlinn(aten::vec3(0.7, 0.6, 0.5), 200, 0.2));
	scene->add(s_blinn);

	auto s_ggx = new aten::sphere(aten::vec3(-3, 0, 0), 1.0, new aten::MicrofacetGGX(aten::vec3(0.7, 0.6, 0.5), 0.2, 0.2));
	scene->add(s_ggx);

	auto s_beckman = new aten::sphere(aten::vec3(+1, 0, 0), 1.0, new aten::MicrofacetBeckman(aten::vec3(0.7, 0.6, 0.5), 0.2, 0.2));
	scene->add(s_beckman);

	auto s_glass = new aten::sphere(aten::vec3(+3, 0, 0), 1.0, new aten::specular(aten::vec3(0.7, 0.6, 0.5)));
	scene->add(s_glass);
}

void MtrlTestScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(0, 0, 13);
	at = aten::vec3(0, 0, 0);
	fov = 30;
}

/////////////////////////////////////////////////////

void ObjectScene::makeScene(aten::scene* scene)
{
	aten::AssetManager::registerMtrl(
		"m1",
		new aten::MicrofacetBlinn(aten::vec3(0.7, 0.6, 0.5), 200, 0.2));

	aten::AssetManager::registerMtrl(
		"Material.001",
		new aten::MicrofacetBlinn(aten::vec3(0.7, 0.6, 0.5), 200, 0.2));

	auto obj = aten::ObjLoader::load("../../asset/suzanne/suzanne.obj");
	//auto obj = aten::ObjLoader::load("../../asset/teapot.obj");

	aten::mat4 mtxL2W;
	mtxL2W.asRotateByZ(Deg2Rad(45));

	aten::mat4 mm;
	mm.asTrans(aten::vec3(-1, 0, 0));

	mtxL2W = mtxL2W * mm;

	auto instance = new aten::instance<aten::object>(obj, mtxL2W);

	scene->add(instance);
}

void ObjectScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(0.0, 0.0, 10.0);
	//pos = aten::vec3(0.0, 0.0, 60.0);
	at = aten::vec3(0.0, 0.0, 0.0);
	fov = 30;
}

/////////////////////////////////////////////////////

void PointLightScene::makeScene(aten::scene* scene)
{
	auto emit = new aten::emissive(aten::vec3(36.0, 36.0, 36.0));

	auto light = new aten::sphere(
		aten::vec3(50.0, 90.0, 81.6),
		15.0,
		emit);

	double r = 1e5;

	auto floor = new aten::sphere(
		aten::vec3(0, -r, 0),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

	// óŒãÖ.
	auto green = new aten::sphere(
		aten::vec3(65, 20, 20),
		20,
		new aten::lambert(aten::vec3(0.25, 0.75, 0.25)));

	//scene->add(light);
	scene->add(floor);
	scene->add(green);

	aten::Light* l = new aten::PointLight(aten::vec3(50.0, 90.0, 81.6), aten::vec3(36.0, 36.0, 36.0), 0, 0.1, 0);
	//aten::Light* l = new aten::AreaLight(light, emit->color());

	scene->addLight(l);
}

void PointLightScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(50.0, 52.0, 295.6);
	at = aten::vec3(50.0, 40.8, 119.0);
	fov = 30;
}

/////////////////////////////////////////////////////

void DirectionalLightScene::makeScene(aten::scene* scene)
{
	auto emit = new aten::emissive(aten::vec3(36.0, 36.0, 36.0));

	auto light = new aten::sphere(
		aten::vec3(50.0, 90.0, 81.6),
		15.0,
		emit);

	double r = 1e5;

	auto floor = new aten::sphere(
		aten::vec3(0, -r, 0),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

	// óŒãÖ.
	auto green = new aten::sphere(
		aten::vec3(65, 20, 20),
		20,
		new aten::lambert(aten::vec3(0.25, 0.75, 0.25)));

	//scene->add(light);
	scene->add(floor);
	scene->add(green);

	aten::Light* l = new aten::DirectionalLight(aten::vec3(1, -1, 1), aten::vec3(36.0, 36.0, 36.0));
	//aten::Light* l = new aten::AreaLight(light, emit->color());

	scene->addLight(l);
}

void DirectionalLightScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(50.0, 52.0, 295.6);
	at = aten::vec3(50.0, 40.8, 119.0);
	fov = 30;
}

/////////////////////////////////////////////////////

void SpotLightScene::makeScene(aten::scene* scene)
{
	double r = 1e5;

	auto floor = new aten::sphere(
		aten::vec3(0, -r, 0),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

	// óŒãÖ.
	auto green = new aten::sphere(
		aten::vec3(65, 20, 20),
		20,
		new aten::lambert(aten::vec3(0.25, 0.75, 0.25)));

	auto red = new aten::sphere(
		aten::vec3(25, 20, 20),
		20,
		new aten::lambert(aten::vec3(0.75, 0.25, 0.25)));

	//scene->add(light);
	scene->add(floor);
	scene->add(green);
	scene->add(red);

	aten::Light* l = new aten::SpotLight(
		aten::vec3(65, 90, 20),
		aten::vec3(0, -1, 0),
		aten::vec3(36.0, 36.0, 36.0), 
		0, 0.1, 0,
		Deg2Rad(30), 
		Deg2Rad(60),
		1);
	//aten::Light* l = new aten::AreaLight(light, emit->color());

	scene->addLight(l);
}

void SpotLightScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(50.0, 52.0, 295.6);
	at = aten::vec3(50.0, 40.8, 119.0);
	fov = 30;
}

/////////////////////////////////////////////////////

void ManyLightScene::makeScene(aten::scene* scene)
{
	auto s = new aten::sphere(aten::vec3(0, -1000, 0), 1000, new aten::lambert(aten::vec3(0.8, 0.8, 0.8)));
	scene->add(s);

#if 1
	int i = 1;
	for (int x = -5; x < 5; x++) {
		for (int z = -5; z < 5; z++) {
			auto choose_mtrl = aten::drand48();

			aten::vec3 center = aten::vec3(
				x + 0.9 * aten::drand48(),
				0.2,
				z + 0.9 * aten::drand48());

			if (length(center - aten::vec3(4, 0.2, 0)) > 0.9) {
				if (choose_mtrl < 0.8) {
					// lambert
					s = new aten::sphere(
						center,
						0.2,
						new aten::lambert(aten::vec3(aten::drand48() * aten::drand48(), aten::drand48() * aten::drand48(), aten::drand48() * aten::drand48())));
				}
				else if (choose_mtrl < 0.95) {
					// specular
					s = new aten::sphere(
						center,
						0.2,
						new aten::specular(aten::vec3(0.5 * (1 + aten::drand48()), 0.5 * (1 + aten::drand48()), 0.5 * (1 + aten::drand48()))));
				}
				else {
					// glass
					s = new aten::sphere(center, 0.2, new aten::refraction(aten::vec3(1), 1.5));
				}

				scene->add(s);
			}
		}
	}
#endif

	s = new aten::sphere(aten::vec3(0, 1, 0), 1.0, new aten::refraction(aten::vec3(1), 1.5));
	scene->add(s);

	s = new aten::sphere(aten::vec3(-4, 1, 0), 1.0, new aten::lambert(aten::vec3(0.8, 0.2, 0.1)));
	scene->add(s);

	s = new aten::sphere(aten::vec3(4, 1, 0), 1.0, new aten::specular(aten::vec3(0.7, 0.6, 0.5)));
	scene->add(s);

	aten::Light* dir = new aten::DirectionalLight(aten::vec3(-1, -1, -1), aten::vec3(0.5, 0.5, 0.5));
	aten::Light* point = new aten::PointLight(aten::vec3(0, 10, -1), aten::vec3(0.0, 0.0, 1.0), 0, 0.5, 0.02);
	aten::Light* spot = new aten::SpotLight(
		aten::vec3(0, 5, 0),
		aten::vec3(0, -1, 0),
		//aten::vec3(0.2, 0.2, 0.2),
		aten::vec3(0.0, 1.0, 0.0),
		0, 0.1, 0,
		Deg2Rad(30),
		Deg2Rad(60),
		1);

	scene->addLight(dir);
	scene->addLight(spot);
	scene->addLight(point);
}

void ManyLightScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(13, 2, 3);
	at = aten::vec3(0, 0, 0);
	fov = 30;
}

/////////////////////////////////////////////////////

void TexturesScene::makeScene(aten::scene* scene)
{
	auto albedo = aten::ImageLoader::load("../../asset/pbr_textures/Brick_baked/T_Brick_Baked_D.tga");
	auto nml = aten::ImageLoader::load("../../asset/pbr_textures/Brick_baked/T_Brick_Baked_N.tga");
	auto rough = aten::ImageLoader::load("../../asset/pbr_textures/Brick_baked/T_Brick_Baked_R.tga");
	auto nml_2 = aten::ImageLoader::load("../../asset/normalmap.png");
	aten::vec3 clr = aten::vec3(1, 1, 1);

	auto s_blinn = new aten::sphere(aten::vec3(-3, 0, 0), 1.0, new aten::MicrofacetBlinn(clr, 200, 0.2, albedo, nml));
	scene->add(s_blinn);
#if 1
	auto s_ggx = new aten::sphere(aten::vec3(-1, 0, 0), 1.0, new aten::MicrofacetGGX(clr, 0.2, 0.2, albedo, nml, rough));
	scene->add(s_ggx);

	auto s_beckman = new aten::sphere(aten::vec3(+1, 0, 0), 1.0, new aten::MicrofacetBeckman(clr, 0.2, 0.2, albedo, nml, rough));
	scene->add(s_beckman);

	auto s_lambert = new aten::sphere(aten::vec3(+3, 0, 0), 1.0, new aten::lambert(clr, albedo, nml));
	scene->add(s_lambert);

	auto s_spec = new aten::sphere(aten::vec3(-3, +2, 0), 1.0, new aten::specular(clr, nullptr, nml_2));
	scene->add(s_spec);

	auto s_ref = new aten::sphere(aten::vec3(-1, +2, 0), 1.0, new aten::specular(clr, nullptr, nml_2));
	scene->add(s_ref);
#endif
}

void TexturesScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(0, 0, 13);
	at = aten::vec3(0, 0, 0);
	fov = 30;
}

/////////////////////////////////////////////////////

void HideLightScene::makeScene(aten::scene* scene)
{
	auto emit = new aten::emissive(aten::vec3(64.0, 64.0, 64.0));

	auto light = new aten::sphere(
		aten::vec3(50.0, 90.0, 81.6),
		15.0,
		emit);

	real cubeheight = 10;

	auto cube = new aten::cube(
		aten::vec3(50.0, 75.0 - cubeheight, 81.6),
		60, cubeheight, 60,
		new aten::lambert(aten::vec3(0.5, 0.5, 0.5)));

	double r = 1e3;

	auto left = new aten::sphere(
		aten::vec3(r + 1, 40.8, 81.6),
		r,
		new aten::lambert(aten::vec3(0.75f, 0.25f, 0.25f)));

	auto right = new aten::sphere(
		aten::vec3(-r + 99, 40.8, 81.6),
		r,
		new aten::lambert(aten::vec3(0.25, 0.25, 0.75)));

	auto wall = new aten::sphere(
		aten::vec3(50, 40.8, r),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

	auto floor = new aten::sphere(
		aten::vec3(50, r, 81.6),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

	auto ceil = new aten::sphere(
		aten::vec3(50, -r + 81.6, 81.6),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

	//auto tex = aten::ImageLoader::load("../../asset/earth.bmp");

	// óŒãÖ.
	auto green = new aten::sphere(
		aten::vec3(65, 20, 20),
		20,
		new aten::lambert(aten::vec3(0.25, 0.75, 0.25)));
	//new aten::lambert(aten::vec3(1, 1, 1), tex));

	// ãæ.
	auto mirror = new aten::sphere(
		aten::vec3(27, 16.5, 47),
		16.5,
		new aten::specular(aten::vec3(0.99, 0.99, 0.99)));

	// ÉKÉâÉX.
	auto glass = new aten::sphere(
		aten::vec3(77, 16.5, 78),
		16.5,
		new aten::refraction(aten::vec3(0.99, 0.99, 0.99), 1.5));

#if 1
	scene->add(light);
	scene->add(cube);
#if 1
	scene->add(left);
	scene->add(right);
	scene->add(wall);
	scene->add(floor);
	scene->add(ceil);
	scene->add(green);
	scene->add(mirror);
	scene->add(glass);
#endif

	aten::Light* l = new aten::AreaLight(light, emit->color());

	scene->addLight(l);
#endif
}

void HideLightScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(50.0, 52.0, 295.6);
	at = aten::vec3(50.0, 40.8, 119.0);
	fov = 30;
}

/////////////////////////////////////////////////////

void DisneyMaterialTestScene::makeScene(aten::scene* scene)
{
	{
		aten::MaterialParameter param;
		param.baseColor = aten::vec3(0.82, 0.67, 0.16);
		param.roughness = 0.3;
		param.specular = 0.5;
		param.metallic = 0.5;

		auto m = new aten::DisneyBRDF(param);
		auto s = new aten::sphere(aten::vec3(0, 0, 0), 1.0, m);
		scene->add(s);
	}

	{
		auto m = new aten::lambert(aten::vec3(0.82, 0.67, 0.16));
		auto s = new aten::sphere(aten::vec3(-3, 0, 0), 1.0, m);
		scene->add(s);
	}

	aten::Light* dir = new aten::DirectionalLight(aten::vec3(-1, -1, -1), aten::vec3(0.5, 0.5, 0.5));
	//scene->addLight(dir);

#if 0
	{
		aten::DisneyBRDF::Parameter param;
		param.sheen = 0.5;

		auto m = new aten::DisneyBRDF(param);
		auto s = new aten::sphere(aten::vec3(-1, 0, 0), 1.0, m);
		scene->add(s);
	}

	{
		aten::DisneyBRDF::Parameter param;
		param.anisotropic = 0.5;

		auto m = new aten::DisneyBRDF(param);
		auto s = new aten::sphere(aten::vec3(+1, 0, 0), 1.0, m);
		scene->add(s);
	}

	{
		aten::DisneyBRDF::Parameter param;
		param.subsurface = 0.5;

		auto m = new aten::DisneyBRDF(param);
		auto s = new aten::sphere(aten::vec3(+3, 0, 0), 1.0, m);
		scene->add(s);
	}
#endif
}

void DisneyMaterialTestScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(0, 0, 13);
	at = aten::vec3(0, 0, 0);
	fov = 30;
}

/////////////////////////////////////////////////////

void LayeredMaterialTestScene::makeScene(aten::scene* scene)
{
	auto spec = new aten::MicrofacetBlinn(aten::vec3(1, 1, 1), 200, 0.8);
	auto diff = new aten::lambert(aten::vec3(0.7, 0.0, 0.0));

	auto layer = new aten::LayeredBSDF();
	layer->add(spec);
	layer->add(diff);

	auto s_layer = new aten::sphere(aten::vec3(-3, 0, 0), 1.0, layer);
	scene->add(s_layer);

	auto s_diff = new aten::sphere(aten::vec3(-1, 0, 0), 1.0, diff);
	scene->add(s_diff);

	auto s_spec = new aten::sphere(aten::vec3(+1, 0, 0), 1.0, new aten::MicrofacetBlinn(aten::vec3(0.7, 0, 0), 200, 0.8));
	scene->add(s_spec);
}

void LayeredMaterialTestScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(0, 0, 13);
	at = aten::vec3(0, 0, 0);
	fov = 30;
}

/////////////////////////////////////////////////////

void ToonShadeTestScene::makeScene(aten::scene* scene)
{
	aten::Light* l = new aten::DirectionalLight(aten::vec3(1, -1, -1), aten::vec3(36.0, 36.0, 36.0));
	auto toonmtrl = new aten::toon(aten::vec3(0.25, 0.75, 0.25), l);

	toonmtrl->setComputeToonShadeFunc([](real c)->real {
		real ret = 1;
		if (c < 0.33) {
			ret = 0;
		}
		else if (c < 0.66) {
			ret = 0.5;
		}
		else if (c < 0.8) {
			ret = 1;
		}
		else {
			ret = 1.5;
		}
		return ret;
	});

	aten::AssetManager::registerMtrl(
		"m1",
		toonmtrl);

	aten::AssetManager::registerMtrl(
		"Material.001",
		toonmtrl);

	auto obj = aten::ObjLoader::load("../../asset/suzanne/suzanne.obj");
	//auto obj = aten::ObjLoader::load("../../asset/teapot.obj");

	auto instance = new aten::instance<aten::object>(obj);

	scene->add(instance);

	aten::mat4 mtxL2W;

#if 1
	mtxL2W.asTrans(aten::vec3(-2.5, 0, 0));

	// ãæ.
	auto mirror = new aten::sphere(
		1,
		new aten::specular(aten::vec3(0.99, 0.99, 0.99)));
	scene->add(new aten::instance<aten::sphere>(mirror, mtxL2W));
#endif

#if 1
	mtxL2W.asTrans(aten::vec3(2.5, 0, 0));

	auto s_ggx = new aten::sphere(
		1.0, 
		new aten::MicrofacetGGX(aten::vec3(0.7, 0.6, 0.5), 0.2, 0.2));
	scene->add(new aten::instance<aten::sphere>(s_ggx, mtxL2W));
#endif

#if 1
	mtxL2W.asTrans(aten::vec3(0, -1, 2));

	// ÉKÉâÉX.
	auto glass = new aten::sphere(
		0.5,
		new aten::refraction(aten::vec3(0.99, 0.99, 0.99), 1.5, true));
	scene->add(new aten::instance<aten::sphere>(glass, mtxL2W));
#endif

	scene->addLight(l);
}

void ToonShadeTestScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(0.0, 0.0, 10.0);
	//pos = aten::vec3(0.0, 0.0, 60.0);
	at = aten::vec3(0.0, 0.0, 0.0);
	fov = 30;
}

/////////////////////////////////////////////////////

void ObjCornellBoxScene::makeScene(aten::scene* scene)
{
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

	auto emit = new aten::emissive(aten::vec3(36, 33, 24));
	aten::AssetManager::registerMtrl(
		"light",
		emit);

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
	aten::ObjLoader::load(objs, "../../asset/cornellbox/orig.obj");

	auto light = new aten::instance<aten::object>(
		objs[0],
		aten::vec3(0),
		aten::vec3(0),
		aten::vec3(1));
	scene->add(light);

	g_movableObj = light;

	auto areaLight = new aten::AreaLight(light, emit->param().baseColor);
	scene->addLight(areaLight);

	for (int i = 1; i < objs.size(); i++) {
		auto box = new aten::instance<aten::object>(objs[i], aten::mat4::Identity);
		scene->add(box);
	}
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

void SponzaScene::makeScene(aten::scene* scene)
{
	auto SP_LUK = aten::ImageLoader::load("../../asset/sponza/sp_luk.JPG");
	auto SP_LUK_nml = aten::ImageLoader::load("../../asset/sponza/sp_luk-nml.png");

	auto _00_SKAP = aten::ImageLoader::load("../../asset/sponza/00_skap.JPG");

	auto _01_STUB = aten::ImageLoader::load("../../asset/sponza/01_STUB.JPG");
	auto _01_STUB_nml = aten::ImageLoader::load("../../asset/sponza/01_STUB-nml.png");

	auto _01_S_BA = aten::ImageLoader::load("../../asset/sponza/01_S_ba.JPG");

	auto _01_ST_KP = aten::ImageLoader::load("../../asset/sponza/01_St_kp.JPG");
	auto _01_ST_KP_nml = aten::ImageLoader::load("../../asset/sponza/01_St_kp-nml.png");

	auto X01_ST = aten::ImageLoader::load("../../asset/sponza/x01_st.JPG");

	auto KAMEN_stup = aten::ImageLoader::load("../../asset/sponza/KAMEN-stup.JPG");

	auto RELJEF = aten::ImageLoader::load("../../asset/sponza/reljef.JPG");
	auto RELJEF_nml = aten::ImageLoader::load("../../asset/sponza/reljef-nml.png");

	auto KAMEN = aten::ImageLoader::load("../../asset/sponza/KAMEN.JPG");
	auto KAMEN_nml = aten::ImageLoader::load("../../asset/sponza/KAMEN-nml.png");

	auto PROZOR1 = aten::ImageLoader::load("../../asset/sponza/prozor1.JPG");

	auto VRATA_KR = aten::ImageLoader::load("../../asset/sponza/vrata_kr.JPG");

	auto VRATA_KO = aten::ImageLoader::load("../../asset/sponza/vrata_ko.JPG");

	aten::AssetManager::registerMtrl(
		"sp_00_luk_mali",
		new aten::lambert(aten::vec3(0.745098, 0.709804, 0.674510), SP_LUK, SP_LUK_nml));
	aten::AssetManager::registerMtrl(
		"sp_svod_kapitel",
		new aten::lambert(aten::vec3(0.713726, 0.705882, 0.658824), _00_SKAP));
	aten::AssetManager::registerMtrl(
		"sp_01_stub_baza_",
		new aten::lambert(aten::vec3(0.784314, 0.784314, 0.784314)));
	aten::AssetManager::registerMtrl(
		"sp_01_stub_kut",
		new aten::lambert(aten::vec3(0.737255, 0.709804, 0.670588), _01_STUB, _01_STUB_nml));
	aten::AssetManager::registerMtrl(
		"sp_00_stup",
		new aten::lambert(aten::vec3(0.737255, 0.709804, 0.670588), _01_STUB, _01_STUB_nml));
	aten::AssetManager::registerMtrl(
		"sp_01_stub_baza",
		new aten::lambert(aten::vec3(0.800000, 0.784314, 0.749020), _01_S_BA));
	aten::AssetManager::registerMtrl(
		"sp_00_luk_mal1",
		new aten::lambert(aten::vec3(0.745098, 0.709804, 0.674510), _01_ST_KP, _01_ST_KP_nml));
	aten::AssetManager::registerMtrl(
		"sp_01_stub",
		new aten::lambert(aten::vec3(0.737255, 0.709804, 0.670588), _01_STUB, _01_STUB_nml));
	aten::AssetManager::registerMtrl(
		"sp_01_stup",
		new aten::lambert(aten::vec3(0.827451, 0.800000, 0.768628), X01_ST));
	aten::AssetManager::registerMtrl(
		"sp_vijenac",
		new aten::lambert(aten::vec3(0.713726, 0.705882, 0.658824), _00_SKAP));
	aten::AssetManager::registerMtrl(
		"sp_00_svod",
		new aten::lambert(aten::vec3(0.941177, 0.866667, 0.737255), KAMEN_stup));	// TODO	specularÇ™Ç†ÇÈÇÃÇ≈ÅAlambertÇ≈Ç»Ç¢.
	aten::AssetManager::registerMtrl(
		"sp_02_reljef",
		new aten::lambert(aten::vec3(0.529412, 0.498039, 0.490196), RELJEF, RELJEF_nml));
	aten::AssetManager::registerMtrl(
		"sp_01_luk_a",
		new aten::lambert(aten::vec3(0.745098, 0.709804, 0.674510), SP_LUK, SP_LUK_nml));
	aten::AssetManager::registerMtrl(
		"sp_zid_vani",
		new aten::lambert(aten::vec3(0.627451, 0.572549, 0.560784), KAMEN, KAMEN_nml));
	aten::AssetManager::registerMtrl(
		"sp_01_stup_baza",
		new aten::lambert(aten::vec3(0.800000, 0.784314, 0.749020), _01_S_BA));
	aten::AssetManager::registerMtrl(
		"sp_00_zid",
		new aten::lambert(aten::vec3(0.627451, 0.572549, 0.560784), KAMEN, KAMEN_nml));
	aten::AssetManager::registerMtrl(
		"sp_00_prozor",
		new aten::lambert(aten::vec3(1.000000, 1.000000, 1.000000), PROZOR1));
	aten::AssetManager::registerMtrl(
		"sp_00_vrata_krug",
		new aten::lambert(aten::vec3(0.784314, 0.784314, 0.784314), VRATA_KR));
	aten::AssetManager::registerMtrl(
		"sp_00_pod",
		new aten::lambert(aten::vec3(0.627451, 0.572549, 0.560784), KAMEN, KAMEN_nml));
	aten::AssetManager::registerMtrl(
		"sp_00_vrata_kock",
		new aten::lambert(aten::vec3(0.784314, 0.784314, 0.784314), VRATA_KO));

	std::vector<aten::object*> objs;

	aten::ObjLoader::load(objs, "../../asset/sponza/sponza.obj");

	objs[0]->importInternalAccelTree("../../asset/sponza/sponza.sbvh");

	auto sponza = new aten::instance<aten::object>(objs[0], aten::mat4::Identity);
	scene->add(sponza);
}

void SponzaScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(0.f, 1.f, 3.f);
	at = aten::vec3(0.f, 1.f, 0.f);
	fov = 45;
}
