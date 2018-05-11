#include "aten.h"

aten::instance<aten::deformable>* getDeformable();
aten::DeformAnimation* getDeformAnm();

class ObjCornellBoxScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at,
		real& fov);
};

class DeformScene {
public:
	static void makeScene(aten::scene* scene);

	static void getCameraPosAndAt(
		aten::vec3& pos,
		aten::vec3& at,
		real& fov);
};

//#define Scene ObjCornellBoxScene
#define Scene DeformScene
