#include "aten.h"
#include "atenscene.h"

class CornellBoxScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at);
};

class RandomScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at);
};

class MtrlTestScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at);
};

class ObjectScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at);
};

class PointLightScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at);
};

class DirectionalLightScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at);
};

class SpotLightScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at);
};

class ManyLightScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at);
};

class TexturesScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at);
};

class HideLightScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at);
};

class DisneyMaterialTestScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at);
};

class LayeredMaterialTestScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at);
};

class ToonShadeTestScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at);
};

#define Scene CornellBoxScene
//#define Scene RandomScene
//#define Scene ObjectScene
//#define Scene MtrlTestScene
//#define Scene PointLightScene
//#define Scene DirectionalLightScene
//#define Scene SpotLightScene
//#define Scene ManyLightScene
//#define Scene TexturesScene
//#define Scene HideLightScene
//#define Scene DisneyMaterialTestScene
//#define Scene LayeredMaterialTestScene
//#define Scene ToonShadeTestScene
