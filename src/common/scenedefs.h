#include "aten.h"
#include "atenscene.h"

aten::instance<aten::object>* getMovableObj();
aten::instance<aten::deformable>* getDeformable();
aten::DeformAnimation* getDeformAnm();

class CornellBoxScene {
public:
    static void makeScene(aten::context& ctxt, aten::scene* scene);

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov);
};

class RandomScene {
public:
    static void makeScene(aten::context& ctxt, aten::scene* scene);

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov);
};

class MtrlTestScene {
public:
    static void makeScene(aten::context& ctxt, aten::scene* scene);

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov);
};

class ObjectScene {
public:
    static void makeScene(aten::context& ctxt, aten::scene* scene);

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov);
};

class PointLightScene {
public:
    static void makeScene(aten::context& ctxt, aten::scene* scene);

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov);

};

class DirectionalLightScene {
public:
    static void makeScene(aten::context& ctxt, aten::scene* scene);

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov);

};

class SpotLightScene {
public:
    static void makeScene(aten::context& ctxt, aten::scene* scene);

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov);

};

class ManyLightScene {
public:
    static void makeScene(aten::context& ctxt, aten::scene* scene);

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov);

};

class TexturesScene {
public:
    static void makeScene(aten::context& ctxt, aten::scene* scene);

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov);
};

class HideLightScene {
public:
    static void makeScene(aten::context& ctxt, aten::scene* scene);

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov);
};

class DisneyMaterialTestScene {
public:
    static void makeScene(aten::context& ctxt, aten::scene* scene);

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov);
};

class LayeredMaterialTestScene {
public:
    static void makeScene(aten::context& ctxt, aten::scene* scene);

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov);
};

class ToonShadeTestScene {
public:
    static void makeScene(aten::context& ctxt, aten::scene* scene);

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov);
};

class ObjCornellBoxScene {
public:
    static void makeScene(aten::context& ctxt, aten::scene* scene);

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov);
};

class SponzaScene {
public:
    static void makeScene(aten::context& ctxt, aten::scene* scene);

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov);
};

class BunnyScene {
public:
    static void makeScene(aten::context& ctxt, aten::scene* scene);

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov);
};

class DeformScene {
public:
    static void makeScene(
        aten::context& ctxt,
        aten::scene* scene);

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov);
};

class DeformInBoxScene {
public:
    static void makeScene(
        aten::context& ctxt,
        aten::scene* scene);

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov);
};

//#define Scene CornellBoxScene
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
//#define Scene ObjCornellBoxScene
#define Scene SponzaScene
//#define Scene BunnyScene
//#define Scene DeformScene
//#define Scene DeformInBoxScene