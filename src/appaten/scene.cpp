#include "scene.h"

void CornellBoxScene::makeScene(aten::scene* scene)
{
	auto light = new aten::sphere(
		aten::vec3(50.0, 90.0, 81.6),
		15.0,
		new aten::emissive(aten::vec3(36.0, 36.0, 36.0)));

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
	scene->add(left);
	scene->add(right);
	scene->add(wall);
	scene->add(floor);
	scene->add(ceil);
	scene->add(green);
	scene->add(mirror);
	scene->add(glass);

	scene->addLight(light);
#endif
}

void CornellBoxScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at)
{
	pos = aten::vec3(50.0, 52.0, 295.6);
	at = aten::vec3(50.0, 40.8, 119.0);
}

/////////////////////////////////////////////////////

aten::real drand48()
{
	return (aten::real)::rand() / RAND_MAX;
}

void RandomScene::makeScene(aten::scene* scene)
{
	auto s = new aten::sphere(aten::vec3(0, -1000, 0), 1000, new aten::lambert(aten::vec3(0.8, 0.8, 0.8)));
	scene->add(s);

	int i = 1;
	for (int x = -11; x < 11; x++) {
		for (int z = -11; z < 11; z++) {
			float choose_mtrl = drand48();

			aten::vec3 center(
				x + 0.9 * drand48(),
				0.2,
				z + 0.9 * drand48());

			if ((center - aten::vec3(4, 0.2, 0)).length() > 0.9) {
				if (choose_mtrl < 0.8) {
					// lambert
					s = new aten::sphere(
						center,
						0.2,
						new aten::lambert(aten::vec3(drand48() * drand48(), drand48() * drand48(), drand48() * drand48())));
				}
				else if (choose_mtrl < 0.95) {
					// specular
					s = new aten::sphere(
						center,
						0.2,
						new aten::specular(aten::vec3(0.5 * (1 + drand48()), 0.5 * (1 + drand48()), 0.5 * (1 + drand48()))));
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
	auto s_blinn = new aten::sphere(aten::vec3(0, 0, 0), 1.0, new aten::MicrofacetBlinn(aten::vec3(0.7, 0.6, 0.5), 200, 0.2));
	scene->add(s_blinn);

	auto s_ggx = new aten::sphere(aten::vec3(-3, 0, 0), 1.0, new aten::MicrofacetGGX(aten::vec3(0.7, 0.6, 0.5), 0.2, 0.2));
	scene->add(s_ggx);

	auto s_beckman = new aten::sphere(aten::vec3(+3, 0, 0), 1.0, new aten::MicrofacetBeckman(aten::vec3(0.7, 0.6, 0.5), 0.2, 0.2));
	scene->add(s_beckman);
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
	//auto obj = aten::ObjLoader::load("../../asset/suzanne.obj");
	auto obj = aten::ObjLoader::load("../../asset/teapot.obj");

	auto instance = new aten::objinstance(obj);

	scene->add(instance);
}

void ObjectScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at)
{
	//pos = aten::vec3(0.0, 0.0, 10.0);
	pos = aten::vec3(0.0, 0.0, 60.0);
	at = aten::vec3(0.0, 0.0, 0.0);
}
