#include "aten.h"

class CornellBoxScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at,
		real& fov);
};

class MtrlTestScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at,
		real& fov);
};

class PointLightScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at,
		real& fov);

};

class DirectionalLightScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at,
		real& fov);

};

class SpotLightScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at,
		real& fov);

};

class ObjCornellBoxScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at,
		real& fov);
};

//#define Scene CornellBoxScene
#define Scene MtrlTestScene
//#define Scene PointLightScene
//#define Scene DirectionalLightScene
//#define Scene SpotLightScene
//#define Scene ObjCornellBoxScene
