#include "aten.h"

class DemoScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at,
		real& fov);
};

#define Scene DemoScene
