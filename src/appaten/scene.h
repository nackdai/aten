#include "aten.h"

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

//#define Scene CornellBoxScene
#define Scene RandomScene
//#define Scene ObjectScene
//#define Scene MtrlTestScene