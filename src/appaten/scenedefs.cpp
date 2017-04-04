#include "scenedefs.h"

void CornellBoxScene::makeScene(aten::scene* scene)
{
	auto emit = new aten::emissive(aten::vec3(36, 36, 36));
	//auto emit = new aten::emissive(aten::vec3(3, 3, 3));

	auto light = new aten::sphere(
		aten::vec3(50.0, 90.0, 81.6),
		15.0,
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
	// —Î‹….
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
	// ‹¾.
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

#if DEFALT
	// ƒKƒ‰ƒX.
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
		emit);

	aten::AssetManager::registerMtrl(
		"Material.001",
		//new aten::MicrofacetBlinn(aten::vec3(0.7, 0.6, 0.5), 200, 0.2));
		emit);

	auto obj = aten::ObjLoader::load("../../asset/suzanne.obj");
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

#if DEFALT
	scene->add(light);
#endif
	scene->add(left);
	scene->add(right);
	scene->add(wall);
	scene->add(floor);
	scene->add(ceil);
	scene->add(green);
	scene->add(mirror);
	scene->add(glass);

#if DEFALT
	aten::Light* l = new aten::AreaLight(light, emit->color());
#else
	aten::Light* l = new aten::AreaLight(glass, emit->color());
#endif

	scene->addLight(l);
}

void CornellBoxScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at)
{
	pos = aten::vec3(50.0, 52.0, 295.6);
	at = aten::vec3(50.0, 40.8, 119.0);
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

			aten::vec3 center(
				x + 0.9 * aten::drand48(),
				0.2,
				z + 0.9 * aten::drand48());

			if ((center - aten::vec3(4, 0.2, 0)).length() > 0.9) {
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
	aten::vec3& at)
{
	pos = aten::vec3(13, 2, 3);
	at = aten::vec3(0, 0, 0);
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
	aten::vec3& at)
{
	pos = aten::vec3(0, 0, 13);
	at = aten::vec3(0, 0, 0);
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

	auto obj = aten::ObjLoader::load("../../asset/suzanne.obj");
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
	aten::vec3& at)
{
	pos = aten::vec3(0.0, 0.0, 10.0);
	//pos = aten::vec3(0.0, 0.0, 60.0);
	at = aten::vec3(0.0, 0.0, 0.0);
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

	// —Î‹….
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
	aten::vec3& at)
{
	pos = aten::vec3(50.0, 52.0, 295.6);
	at = aten::vec3(50.0, 40.8, 119.0);
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

	// —Î‹….
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
	aten::vec3& at)
{
	pos = aten::vec3(50.0, 52.0, 295.6);
	at = aten::vec3(50.0, 40.8, 119.0);
}

/////////////////////////////////////////////////////

void SpotLightScene::makeScene(aten::scene* scene)
{
	double r = 1e5;

	auto floor = new aten::sphere(
		aten::vec3(0, -r, 0),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

	// —Î‹….
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
	aten::vec3& at)
{
	pos = aten::vec3(50.0, 52.0, 295.6);
	at = aten::vec3(50.0, 40.8, 119.0);
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

			aten::vec3 center(
				x + 0.9 * aten::drand48(),
				0.2,
				z + 0.9 * aten::drand48());

			if ((center - aten::vec3(4, 0.2, 0)).length() > 0.9) {
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
	aten::vec3& at)
{
	pos = aten::vec3(13, 2, 3);
	at = aten::vec3(0, 0, 0);
}

/////////////////////////////////////////////////////

void TexturesScene::makeScene(aten::scene* scene)
{
	auto albedo = aten::ImageLoader::load("../../asset/pbr_textures/Brick_baked/T_Brick_Baked_D.tga");
	auto nml = aten::ImageLoader::load("../../asset/pbr_textures/Brick_baked/T_Brick_Baked_N.tga");
	auto rough = aten::ImageLoader::load("../../asset/pbr_textures/Brick_baked/T_Brick_Baked_R.tga");
	auto nml_2 = aten::ImageLoader::load("../../asset/normalmap.png");
	aten::vec3 clr(1, 1, 1);

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
	aten::vec3& at)
{
	pos = aten::vec3(0, 0, 13);
	at = aten::vec3(0, 0, 0);
}

/////////////////////////////////////////////////////

void HideLightScene::makeScene(aten::scene* scene)
{
	auto emit = new aten::emissive(aten::vec3(64.0, 64.0, 64.0));

	auto light = new aten::sphere(
		aten::vec3(50.0, 90.0, 81.6),
		15.0,
		emit);

	aten::real cubeheight = 10;

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

	// —Î‹….
	auto green = new aten::sphere(
		aten::vec3(65, 20, 20),
		20,
		new aten::lambert(aten::vec3(0.25, 0.75, 0.25)));
	//new aten::lambert(aten::vec3(1, 1, 1), tex));

	// ‹¾.
	auto mirror = new aten::sphere(
		aten::vec3(27, 16.5, 47),
		16.5,
		new aten::specular(aten::vec3(0.99, 0.99, 0.99)));

	// ƒKƒ‰ƒX.
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
	aten::vec3& at)
{
	pos = aten::vec3(50.0, 52.0, 295.6);
	at = aten::vec3(50.0, 40.8, 119.0);
}

/////////////////////////////////////////////////////

void DisneyMaterialTestScene::makeScene(aten::scene* scene)
{
	{
		aten::DisneyBRDF::Parameter param;
		param.roughness = 0.5;
		param.specular = 1.0;
		param.metallic = 1.0;

		auto m = new aten::DisneyBRDF(param);
		auto s = new aten::sphere(aten::vec3(-3, 0, 0), 1.0, m);
		scene->add(s);
	}

	{
		auto m = new aten::lambert(aten::vec3(0.82, 0.67, 0.16));
		auto s = new aten::sphere(aten::vec3(-1, 0, 0), 1.0, m);
		scene->add(s);
	}

	aten::Light* dir = new aten::DirectionalLight(aten::vec3(-1, -1, -1), aten::vec3(0.5, 0.5, 0.5));
	scene->addLight(dir);

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
	aten::vec3& at)
{
	pos = aten::vec3(0, 0, 13);
	at = aten::vec3(0, 0, 0);
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
	aten::vec3& at)
{
	pos = aten::vec3(0, 0, 13);
	at = aten::vec3(0, 0, 0);
}

/////////////////////////////////////////////////////

void ToonShadeTestScene::makeScene(aten::scene* scene)
{
	aten::Light* l = new aten::DirectionalLight(aten::vec3(1, -1, -1), aten::vec3(36.0, 36.0, 36.0));
	auto toonmtrl = new aten::toon(aten::vec3(0.25, 0.75, 0.25), l);

	toonmtrl->setComputeToonShadeFunc([](aten::real c)->aten::real {
		aten::real ret = 1;
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

	auto obj = aten::ObjLoader::load("../../asset/suzanne.obj");
	//auto obj = aten::ObjLoader::load("../../asset/teapot.obj");

	auto instance = new aten::instance<aten::object>(obj);

	scene->add(instance);

	aten::mat4 mtxL2W;

#if 1
	mtxL2W.asTrans(aten::vec3(-2.5, 0, 0));

	// ‹¾.
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

	// ƒKƒ‰ƒX.
	auto glass = new aten::sphere(
		0.5,
		new aten::refraction(aten::vec3(0.99, 0.99, 0.99), 1.5, true));
	scene->add(new aten::instance<aten::sphere>(glass, mtxL2W));
#endif

	scene->addLight(l);
}

void ToonShadeTestScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at)
{
	pos = aten::vec3(0.0, 0.0, 10.0);
	//pos = aten::vec3(0.0, 0.0, 60.0);
	at = aten::vec3(0.0, 0.0, 0.0);
}
