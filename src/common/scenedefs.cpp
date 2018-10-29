#include "scenedefs.h"

static aten::instance<aten::object>* g_movableObj = nullptr;
static aten::instance<aten::deformable>* s_deformMdl = nullptr;
static aten::DeformAnimation* s_deformAnm = nullptr;

aten::instance<aten::object>* getMovableObj()
{
    return g_movableObj;
}

aten::instance<aten::deformable>* getDeformable()
{
    return s_deformMdl;
}

aten::DeformAnimation* getDeformAnm()
{
    return s_deformAnm;
}

static aten::material* createMaterial(
    aten::context& ctxt, 
    aten::MaterialType type,
    const aten::vec3& albedo,
    aten::texture* albedoMap,
    aten::texture* normalMap)
{
    aten::MaterialParameter param;
    param.baseColor = albedo;

    auto mtrl = ctxt.createMaterialWithMaterialParameter(
        type,
        param,
        albedoMap,
        normalMap,
        nullptr);

    return mtrl;
}

static aten::material* createMaterial(
    aten::context& ctxt,
    aten::MaterialType type,
    const aten::vec3& albedo)
{
    aten::MaterialParameter param;
    param.baseColor = albedo;

    return ctxt.createMaterialWithMaterialParameter(
        type,
        param,
        nullptr, nullptr, nullptr);
}

static aten::material* createMaterialWithParamter(
    aten::context& ctxt,
    aten::MaterialType type, 
    const aten::MaterialParameter& param)
{
    return ctxt.createMaterialWithMaterialParameter(
        type,
        param,
        nullptr, nullptr, nullptr);
}

static aten::material* createMaterialWithParamter(
    aten::context& ctxt,
    aten::MaterialType type,
    const aten::MaterialParameter& param,
    aten::texture* albedoMap,
    aten::texture* normalMap,
    aten::texture* roughnessMap)
{
    return ctxt.createMaterialWithMaterialParameter(
        type,
        param,
        albedoMap, 
        normalMap,
        roughnessMap);
}

void CornellBoxScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    auto emit = createMaterial(ctxt, aten::MaterialType::Emissive, aten::vec3(36, 36, 36));

    auto light = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(50.0, 75.0, 81.6),
        5.0,
        emit);

    double r = 1e3;

    auto left = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(r + 1, 40.8, 81.6),
        r,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.75f, 0.25f, 0.25f)));

    auto right = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(-r + 99, 40.8, 81.6),
        r,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.25, 0.25, 0.75)));

    auto wall = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(50, 40.8, r),
        r,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.75, 0.75, 0.75)));

    auto floor = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(50, r, 81.6),
        r,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.75, 0.75, 0.75)));

    auto ceil = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(50, -r + 81.6, 81.6),
        r,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.75, 0.75, 0.75)));

    //auto tex = aten::ImageLoader::load("../../asset/earth.bmp");

#define DEFALT    (1)

#if DEFALT
    // óŒãÖ.
    auto green = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(65, 20, 20),
        20,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.25, 0.75, 0.25)));
        //createMaterial(ctxt, aten::vec3(1, 1, 1), tex));
#else
    auto green = aten::TransformableFactory::createSphere(
        aten::vec3(65, 20, 20),
        20,
        new aten::MicrofacetGGX(aten::vec3(0.7, 0.6, 0.5), 0.2, 0.2));
#endif

#if DEFALT
    // ãæ.
    auto mirror = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(27, 16.5, 47),
        16.5,
        createMaterial(ctxt, aten::MaterialType::Specular, aten::vec3(0.99, 0.99, 0.99)));
#else
    auto spec = new aten::MicrofacetBlinn(aten::vec3(1, 1, 1), 200, 0.8);
    auto diff = createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.0, 0.7, 0.0));

    auto layer = new aten::LayeredBSDF();
    layer->add(spec);
    layer->add(diff);

    auto mirror = aten::TransformableFactory::createSphere(
        aten::vec3(27, 16.5, 47),
        16.5,
        layer);
#endif

//#if DEFALT
#if 0
    // ÉKÉâÉX.
    auto glass = aten::TransformableFactory::createSphere(
        aten::vec3(77, 16.5, 78),
        16.5,
        new aten::refraction(aten::vec3(0.99, 0.99, 0.99), 1.5));
#elif 0
    auto glass = aten::TransformableFactory::createSphere(
        aten::vec3(77, 16.5, 78),
        5,
        emit);
#else
    aten::AssetManager::registerMtrl(
        "m1",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.2, 0.2, 0.7)));

    aten::AssetManager::registerMtrl(
        "Material.001",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.2, 0.2, 0.7)));

    auto obj = aten::ObjLoader::load("../../asset/suzanne/suzanne.obj", ctxt);
    //auto obj = aten::ObjLoader::load("../../asset/teapot.obj");

    aten::mat4 mtxL2W;
    mtxL2W.asRotateByY(Deg2Rad(-25));

    aten::mat4 mtxT;
    mtxT.asTrans(aten::vec3(77, 16.5, 78));

    aten::mat4 mtxS;
    mtxS.asScale(10);

    mtxL2W = mtxT * mtxL2W * mtxS;

    auto glass = aten::TransformableFactory::createInstance<aten::object>(ctxt, obj, mtxL2W);
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

void RandomScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    auto s = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(0, -1000, 0), 
        1000, 
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.8, 0.8, 0.8)));
    scene->add(s);

    aten::MaterialParameter mtrlParam;

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
                    s = aten::TransformableFactory::createSphere(
                        ctxt,
                        center,
                        0.2,
                        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(aten::drand48() * aten::drand48(), aten::drand48() * aten::drand48(), aten::drand48() * aten::drand48())));
                }
                else if (choose_mtrl < 0.95) {
                    // specular
                    s = aten::TransformableFactory::createSphere(
                        ctxt,
                        center,
                        0.2,
                        createMaterial(ctxt, aten::MaterialType::Specular, aten::vec3(0.5 * (1 + aten::drand48()), 0.5 * (1 + aten::drand48()), 0.5 * (1 + aten::drand48()))));
                }
                else {
                    // glass
                    mtrlParam.baseColor = aten::vec3(1);
                    mtrlParam.ior = 1.5;

                    s = aten::TransformableFactory::createSphere(
                        ctxt,
                        center, 
                        0.2, 
                        createMaterialWithParamter(ctxt, aten::MaterialType::Refraction, mtrlParam));
                }

                scene->add(s);
            }
        }
    }

    mtrlParam.baseColor = aten::vec3(1);
    mtrlParam.ior = 1.5;
    s = aten::TransformableFactory::createSphere(ctxt, aten::vec3(0, 1, 0), 1.0, createMaterialWithParamter(ctxt, aten::MaterialType::Refraction, mtrlParam));
    scene->add(s);

    s = aten::TransformableFactory::createSphere(ctxt, aten::vec3(-4, 1, 0), 1.0, createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.4, 0.2, 0.1)));
    scene->add(s);

    s = aten::TransformableFactory::createSphere(ctxt, aten::vec3(4, 1, 0), 1.0, createMaterial(ctxt, aten::MaterialType::Specular, aten::vec3(0.7, 0.6, 0.5)));
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

void MtrlTestScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    aten::MaterialParameter mtrlParam;

    mtrlParam.baseColor = aten::vec3(0.7, 0.6, 0.5);
    mtrlParam.shininess = 200;
    mtrlParam.ior = 0.2;
    auto s_blinn = aten::TransformableFactory::createSphere(ctxt, aten::vec3(-1, 0, 0), 1.0, createMaterialWithParamter(ctxt, aten::MaterialType::Blinn, mtrlParam));
    scene->add(s_blinn);

    mtrlParam.baseColor = aten::vec3(0.7, 0.6, 0.5);
    mtrlParam.roughness = 0.2;
    mtrlParam.ior = 0.2;
    auto s_ggx = aten::TransformableFactory::createSphere(ctxt, aten::vec3(-3, 0, 0), 1.0, createMaterialWithParamter(ctxt, aten::MaterialType::GGX, mtrlParam));
    scene->add(s_ggx);

    mtrlParam.baseColor = aten::vec3(0.7, 0.6, 0.5);
    mtrlParam.roughness = 0.2;
    mtrlParam.ior = 0.2;
    auto s_beckman = aten::TransformableFactory::createSphere(ctxt, aten::vec3(+1, 0, 0), 1.0, createMaterialWithParamter(ctxt, aten::MaterialType::Beckman, mtrlParam));
    scene->add(s_beckman);

    auto s_glass = aten::TransformableFactory::createSphere(ctxt, aten::vec3(+3, 0, 0), 1.0, createMaterial(ctxt, aten::MaterialType::Specular, aten::vec3(0.7, 0.6, 0.5)));
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

void ObjectScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    aten::MaterialParameter mtrlParam;
    mtrlParam.baseColor = aten::vec3(0.7, 0.6, 0.5);
    mtrlParam.shininess = 200;
    mtrlParam.ior = 0.2;

    aten::AssetManager::registerMtrl(
        "m1",
        createMaterialWithParamter(ctxt, aten::MaterialType::Blinn, mtrlParam));

    aten::AssetManager::registerMtrl(
        "Material.001",
        createMaterialWithParamter(ctxt, aten::MaterialType::Blinn, mtrlParam));

    auto obj = aten::ObjLoader::load("../../asset/suzanne/suzanne.obj", ctxt);
    //auto obj = aten::ObjLoader::load("../../asset/teapot.obj");

    aten::mat4 mtxL2W;
    mtxL2W.asRotateByZ(Deg2Rad(45));

    aten::mat4 mm;
    mm.asTrans(aten::vec3(-1, 0, 0));

    mtxL2W = mtxL2W * mm;

    auto instance = aten::TransformableFactory::createInstance<aten::object>(ctxt, obj, mtxL2W);

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

void PointLightScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    auto emit = createMaterial(ctxt, aten::MaterialType::Emissive, aten::vec3(36.0, 36.0, 36.0));

    auto light = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(50.0, 90.0, 81.6),
        15.0,
        emit);

    double r = 1e5;

    auto floor = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(0, -r, 0),
        r,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.75, 0.75, 0.75)));

    // óŒãÖ.
    auto green = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(65, 20, 20),
        20,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.25, 0.75, 0.25)));

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

void DirectionalLightScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    auto emit = createMaterial(ctxt, aten::MaterialType::Emissive, aten::vec3(36.0, 36.0, 36.0));

    auto light = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(50.0, 90.0, 81.6),
        15.0,
        emit);

    double r = 1e5;

    auto floor = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(0, -r, 0),
        r,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.75, 0.75, 0.75)));

    // óŒãÖ.
    auto green = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(65, 20, 20),
        20,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.25, 0.75, 0.25)));

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

void SpotLightScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    double r = 1e5;

    auto floor = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(0, -r, 0),
        r,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.75, 0.75, 0.75)));

    // óŒãÖ.
    auto green = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(65, 20, 20),
        20,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.25, 0.75, 0.25)));

    auto red = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(25, 20, 20),
        20,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.75, 0.25, 0.25)));

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

void ManyLightScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    auto s = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(0, -1000, 0), 
        1000, 
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.8, 0.8, 0.8)));
    scene->add(s);

    aten::MaterialParameter mtrlParam;

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
                    s = aten::TransformableFactory::createSphere(
                        ctxt,
                        center,
                        0.2,
                        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(aten::drand48() * aten::drand48(), aten::drand48() * aten::drand48(), aten::drand48() * aten::drand48())));
                }
                else if (choose_mtrl < 0.95) {
                    // specular
                    s = aten::TransformableFactory::createSphere(
                        ctxt,
                        center,
                        0.2,
                        createMaterial(ctxt, aten::MaterialType::Specular, aten::vec3(0.5 * (1 + aten::drand48()), 0.5 * (1 + aten::drand48()), 0.5 * (1 + aten::drand48()))));
                }
                else {
                    // glass
                    mtrlParam.baseColor = aten::vec3(1);
                    mtrlParam.ior = 1.5;

                    s = aten::TransformableFactory::createSphere(ctxt, center, 0.2, createMaterialWithParamter(ctxt, aten::MaterialType::Refraction, mtrlParam));
                }

                scene->add(s);
            }
        }
    }
#endif

    mtrlParam.baseColor = aten::vec3(1);
    mtrlParam.ior = 1.5;
    s = aten::TransformableFactory::createSphere(ctxt, aten::vec3(0, 1, 0), 1.0, createMaterialWithParamter(ctxt, aten::MaterialType::Refraction, mtrlParam));
    scene->add(s);

    s = aten::TransformableFactory::createSphere(ctxt, aten::vec3(-4, 1, 0), 1.0, createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.8, 0.2, 0.1)));
    scene->add(s);

    s = aten::TransformableFactory::createSphere(ctxt, aten::vec3(4, 1, 0), 1.0, createMaterial(ctxt, aten::MaterialType::Specular, aten::vec3(0.7, 0.6, 0.5)));
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

void TexturesScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    auto albedo = aten::ImageLoader::load("../../asset/pbr_textures/Brick_baked/T_Brick_Baked_D.tga", ctxt);
    auto nml = aten::ImageLoader::load("../../asset/pbr_textures/Brick_baked/T_Brick_Baked_N.tga", ctxt);
    auto rough = aten::ImageLoader::load("../../asset/pbr_textures/Brick_baked/T_Brick_Baked_R.tga", ctxt);
    auto nml_2 = aten::ImageLoader::load("../../asset/normalmap.png", ctxt);
    aten::vec3 clr = aten::vec3(1, 1, 1);

    aten::MaterialParameter mtrlParam;

    mtrlParam.baseColor = clr;
    mtrlParam.shininess = 200;
    mtrlParam.ior = 0.2;
    auto s_blinn = aten::TransformableFactory::createSphere(ctxt, aten::vec3(-3, 0, 0), 1.0, createMaterialWithParamter(ctxt, aten::MaterialType::Blinn, mtrlParam, albedo, nml, nullptr));
    scene->add(s_blinn);

#if 1
    mtrlParam.baseColor = clr;
    mtrlParam.roughness = 0.2;
    mtrlParam.ior = 0.2;
    auto s_ggx = aten::TransformableFactory::createSphere(ctxt, aten::vec3(-1, 0, 0), 1.0, createMaterialWithParamter(ctxt, aten::MaterialType::GGX, mtrlParam, albedo, nml, rough));
    scene->add(s_ggx);

    mtrlParam.baseColor = clr;
    mtrlParam.roughness = 0.2;
    mtrlParam.ior = 0.2;
    auto s_beckman = aten::TransformableFactory::createSphere(ctxt, aten::vec3(+1, 0, 0), 1.0, createMaterialWithParamter(ctxt, aten::MaterialType::Beckman, mtrlParam, albedo, nml, rough));
    scene->add(s_beckman);

    mtrlParam.baseColor = clr;
    mtrlParam.roughness = 0.2;
    mtrlParam.ior = 0.2;
    auto s_lambert = aten::TransformableFactory::createSphere(ctxt, aten::vec3(+3, 0, 0), 1.0, createMaterial(ctxt, aten::MaterialType::Lambert, clr, albedo, nml));
    scene->add(s_lambert);

    auto s_spec = aten::TransformableFactory::createSphere(ctxt, aten::vec3(-3, +2, 0), 1.0, createMaterial(ctxt, aten::MaterialType::Specular, clr, nullptr, nml_2));
    scene->add(s_spec);

    auto s_ref = aten::TransformableFactory::createSphere(ctxt, aten::vec3(-1, +2, 0), 1.0, createMaterial(ctxt, aten::MaterialType::Specular, clr, nullptr, nml_2));
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

void HideLightScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    auto emit = createMaterial(ctxt, aten::MaterialType::Emissive, aten::vec3(64.0, 64.0, 64.0));

    auto light = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(50.0, 90.0, 81.6),
        15.0,
        emit);

    real cubeheight = 10;

    auto cube = aten::TransformableFactory::createCube(
        ctxt,
        aten::vec3(50.0, 75.0 - cubeheight, 81.6),
        60, cubeheight, 60,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.5, 0.5, 0.5)));

    double r = 1e3;

    auto left = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(r + 1, 40.8, 81.6),
        r,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.75f, 0.25f, 0.25f)));

    auto right = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(-r + 99, 40.8, 81.6),
        r,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.25, 0.25, 0.75)));

    auto wall = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(50, 40.8, r),
        r,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.75, 0.75, 0.75)));

    auto floor = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(50, r, 81.6),
        r,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.75, 0.75, 0.75)));

    auto ceil = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(50, -r + 81.6, 81.6),
        r,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.75, 0.75, 0.75)));

    //auto tex = aten::ImageLoader::load("../../asset/earth.bmp");

    // óŒãÖ.
    auto green = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(65, 20, 20),
        20,
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.25, 0.75, 0.25)));
    //createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(1, 1, 1), tex));

    // ãæ.
    auto mirror = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(27, 16.5, 47),
        16.5,
        createMaterial(ctxt, aten::MaterialType::Specular, aten::vec3(0.99, 0.99, 0.99)));

    // ÉKÉâÉX.
    aten::MaterialParameter mtrlParam;
    mtrlParam.baseColor = aten::vec3(0.99, 0.99, 0.99);
    mtrlParam.ior = 1.5;
    auto glass = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(77, 16.5, 78),
        16.5,
        createMaterialWithParamter(ctxt, aten::MaterialType::Refraction, mtrlParam));

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

void DisneyMaterialTestScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    {
        aten::MaterialParameter mtrlParam;
        mtrlParam.baseColor = aten::vec3(0.82, 0.67, 0.16);
        mtrlParam.roughness = 0.3;
        mtrlParam.specular = 0.5;
        mtrlParam.metallic = 0.5;

        auto m = createMaterialWithParamter(ctxt, aten::MaterialType::Disney, mtrlParam);;
        auto s = aten::TransformableFactory::createSphere(ctxt, aten::vec3(0, 0, 0), 1.0, m);
        scene->add(s);
    }

    {
        auto m = createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.82, 0.67, 0.16));
        auto s = aten::TransformableFactory::createSphere(ctxt, aten::vec3(-3, 0, 0), 1.0, m);
        scene->add(s);
    }

    aten::Light* dir = new aten::DirectionalLight(aten::vec3(-1, -1, -1), aten::vec3(0.5, 0.5, 0.5));
    //scene->addLight(dir);

#if 0
    {
        aten::DisneyBRDF::Parameter param;
        param.sheen = 0.5;

        auto m = new aten::DisneyBRDF(param);
        auto s = aten::TransformableFactory::createSphere(aten::vec3(-1, 0, 0), 1.0, m);
        scene->add(s);
    }

    {
        aten::DisneyBRDF::Parameter param;
        param.anisotropic = 0.5;

        auto m = new aten::DisneyBRDF(param);
        auto s = aten::TransformableFactory::createSphere(aten::vec3(+1, 0, 0), 1.0, m);
        scene->add(s);
    }

    {
        aten::DisneyBRDF::Parameter param;
        param.subsurface = 0.5;

        auto m = new aten::DisneyBRDF(param);
        auto s = aten::TransformableFactory::createSphere(aten::vec3(+3, 0, 0), 1.0, m);
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

void LayeredMaterialTestScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    aten::MaterialParameter mtrlParam;

    mtrlParam.baseColor = aten::vec3(1, 1, 1);
    mtrlParam.shininess = 200;
    mtrlParam.ior = 0.8;
    auto spec = createMaterialWithParamter(ctxt, aten::MaterialType::Blinn, mtrlParam);
    
    auto diff = createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.7, 0.0, 0.0));

    auto layer = new aten::LayeredBSDF();
    layer->add(spec);
    layer->add(diff);

    auto s_layer = aten::TransformableFactory::createSphere(ctxt, aten::vec3(-3, 0, 0), 1.0, layer);
    scene->add(s_layer);

    auto s_diff = aten::TransformableFactory::createSphere(ctxt, aten::vec3(-1, 0, 0), 1.0, diff);
    scene->add(s_diff);

    mtrlParam.baseColor = aten::vec3(0.7, 0, 0);
    mtrlParam.shininess = 200;
    mtrlParam.ior = 0.8;
    auto s_spec = aten::TransformableFactory::createSphere(ctxt, aten::vec3(+1, 0, 0), 1.0, createMaterialWithParamter(ctxt, aten::MaterialType::Blinn, mtrlParam));
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

void ToonShadeTestScene::makeScene(aten::context& ctxt, aten::scene* scene)
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

    auto obj = aten::ObjLoader::load("../../asset/suzanne/suzanne.obj", ctxt);
    //auto obj = aten::ObjLoader::load("../../asset/teapot.obj");

    auto instance = aten::TransformableFactory::createInstance<aten::object>(ctxt, obj);

    scene->add(instance);

    aten::MaterialParameter mtrlParam;

    aten::mat4 mtxL2W;

#if 1
    mtxL2W.asTrans(aten::vec3(-2.5, 0, 0));

    // ãæ.
    auto mirror = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(0),
        1,
        createMaterial(ctxt, aten::MaterialType::Specular, aten::vec3(0.99, 0.99, 0.99)));
    scene->add(aten::TransformableFactory::createInstance<aten::sphere>(ctxt, mirror, mtxL2W));
#endif

#if 1
    mtxL2W.asTrans(aten::vec3(2.5, 0, 0));

    mtrlParam.baseColor = aten::vec3(0.7, 0.6, 0.5);
    mtrlParam.roughness = 0.2;
    mtrlParam.ior = 0.2;

    auto s_ggx = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(0),
        1.0, 
        createMaterialWithParamter(ctxt, aten::MaterialType::GGX, mtrlParam));
    scene->add(aten::TransformableFactory::createInstance<aten::sphere>(ctxt, s_ggx, mtxL2W));
#endif

#if 1
    mtxL2W.asTrans(aten::vec3(0, -1, 2));

    mtrlParam.baseColor = aten::vec3(0.99, 0.99, 0.99);
    mtrlParam.ior = 1.5;
    mtrlParam.isIdealRefraction = true;

    // ÉKÉâÉX.
    auto glass = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(0),
        0.5,
        createMaterialWithParamter(ctxt, aten::MaterialType::Refraction, mtrlParam));
    scene->add(aten::TransformableFactory::createInstance<aten::sphere>(ctxt, glass, mtxL2W));
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

void ObjCornellBoxScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    aten::AssetManager::registerMtrl(
        "backWall",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.580000, 0.568000, 0.544000)));
    aten::AssetManager::registerMtrl(
        "ceiling",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.580000, 0.568000, 0.544000)));
    aten::AssetManager::registerMtrl(
        "floor",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.580000, 0.568000, 0.544000)));
    aten::AssetManager::registerMtrl(
        "leftWall",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.504000, 0.052000, 0.040000)));

    auto emit = createMaterial(ctxt, aten::MaterialType::Emissive, aten::vec3(36, 33, 24));
    aten::AssetManager::registerMtrl(
        "light",
        emit);

    aten::AssetManager::registerMtrl(
        "rightWall",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.112000, 0.360000, 0.072800)));
    aten::AssetManager::registerMtrl(
        "shortBox",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.580000, 0.568000, 0.544000)));
    aten::AssetManager::registerMtrl(
        "tallBox",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.580000, 0.568000, 0.544000)));

    std::vector<aten::object*> objs;
    aten::ObjLoader::load(objs, "../../asset/cornellbox/orig.obj", ctxt, true);

    auto light = aten::TransformableFactory::createInstance<aten::object>(
        ctxt,
        objs[0],
        aten::vec3(0),
        aten::vec3(0),
        aten::vec3(1));
    scene->add(light);

    g_movableObj = light;

    auto areaLight = new aten::AreaLight(light, emit->param().baseColor);
    scene->addLight(areaLight);

    for (int i = 1; i < objs.size(); i++) {
        auto box = aten::TransformableFactory::createInstance<aten::object>(ctxt, objs[i], aten::mat4::Identity);
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

void SponzaScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    auto SP_LUK = aten::ImageLoader::load("../../asset/sponza/sp_luk.JPG", ctxt);
    auto SP_LUK_nml = aten::ImageLoader::load("../../asset/sponza/sp_luk-nml.png", ctxt);

    auto _00_SKAP = aten::ImageLoader::load("../../asset/sponza/00_skap.JPG", ctxt);

    auto _01_STUB = aten::ImageLoader::load("../../asset/sponza/01_STUB.JPG", ctxt);
    auto _01_STUB_nml = aten::ImageLoader::load("../../asset/sponza/01_STUB-nml.png", ctxt);

    auto _01_S_BA = aten::ImageLoader::load("../../asset/sponza/01_S_ba.JPG", ctxt);

    auto _01_ST_KP = aten::ImageLoader::load("../../asset/sponza/01_St_kp.JPG", ctxt);
    auto _01_ST_KP_nml = aten::ImageLoader::load("../../asset/sponza/01_St_kp-nml.png", ctxt);

    auto X01_ST = aten::ImageLoader::load("../../asset/sponza/x01_st.JPG", ctxt);

    auto KAMEN_stup = aten::ImageLoader::load("../../asset/sponza/KAMEN-stup.JPG", ctxt);

    auto RELJEF = aten::ImageLoader::load("../../asset/sponza/reljef.JPG", ctxt);
    auto RELJEF_nml = aten::ImageLoader::load("../../asset/sponza/reljef-nml.png", ctxt);

    auto KAMEN = aten::ImageLoader::load("../../asset/sponza/KAMEN.JPG", ctxt);
    auto KAMEN_nml = aten::ImageLoader::load("../../asset/sponza/KAMEN-nml.png", ctxt);

    auto PROZOR1 = aten::ImageLoader::load("../../asset/sponza/prozor1.JPG", ctxt);

    auto VRATA_KR = aten::ImageLoader::load("../../asset/sponza/vrata_kr.JPG", ctxt);

    auto VRATA_KO = aten::ImageLoader::load("../../asset/sponza/vrata_ko.JPG", ctxt);

    aten::AssetManager::registerMtrl(
        "sp_00_luk_mali",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.745098, 0.709804, 0.674510), SP_LUK, SP_LUK_nml));
    aten::AssetManager::registerMtrl(
        "sp_svod_kapitel",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.713726, 0.705882, 0.658824), _00_SKAP, nullptr));
    aten::AssetManager::registerMtrl(
        "sp_01_stub_baza_",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.784314, 0.784314, 0.784314)));
    aten::AssetManager::registerMtrl(
        "sp_01_stub_kut",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.737255, 0.709804, 0.670588), _01_STUB, _01_STUB_nml));
    aten::AssetManager::registerMtrl(
        "sp_00_stup",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.737255, 0.709804, 0.670588), _01_STUB, _01_STUB_nml));
    aten::AssetManager::registerMtrl(
        "sp_01_stub_baza",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.800000, 0.784314, 0.749020), _01_S_BA, nullptr));
    aten::AssetManager::registerMtrl(
        "sp_00_luk_mal1",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.745098, 0.709804, 0.674510), _01_ST_KP, _01_ST_KP_nml));
    aten::AssetManager::registerMtrl(
        "sp_01_stub",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.737255, 0.709804, 0.670588), _01_STUB, _01_STUB_nml));
    aten::AssetManager::registerMtrl(
        "sp_01_stup",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.827451, 0.800000, 0.768628), X01_ST, nullptr));
    aten::AssetManager::registerMtrl(
        "sp_vijenac",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.713726, 0.705882, 0.658824), _00_SKAP, nullptr));
    aten::AssetManager::registerMtrl(
        "sp_00_svod",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.941177, 0.866667, 0.737255), KAMEN_stup, nullptr));    // TODO    specularÇ™Ç†ÇÈÇÃÇ≈ÅAlambertÇ≈Ç»Ç¢.
    aten::AssetManager::registerMtrl(
        "sp_02_reljef",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.529412, 0.498039, 0.490196), RELJEF, RELJEF_nml));
    aten::AssetManager::registerMtrl(
        "sp_01_luk_a",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.745098, 0.709804, 0.674510), SP_LUK, SP_LUK_nml));
    aten::AssetManager::registerMtrl(
        "sp_zid_vani",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.627451, 0.572549, 0.560784), KAMEN, KAMEN_nml));
    aten::AssetManager::registerMtrl(
        "sp_01_stup_baza",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.800000, 0.784314, 0.749020), _01_S_BA, nullptr));
    aten::AssetManager::registerMtrl(
        "sp_00_zid",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.627451, 0.572549, 0.560784), KAMEN, KAMEN_nml));
    aten::AssetManager::registerMtrl(
        "sp_00_prozor",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(1.000000, 1.000000, 1.000000), PROZOR1, nullptr));
    aten::AssetManager::registerMtrl(
        "sp_00_vrata_krug",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.784314, 0.784314, 0.784314), VRATA_KR, nullptr));
    aten::AssetManager::registerMtrl(
        "sp_00_pod",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.627451, 0.572549, 0.560784), KAMEN, KAMEN_nml));
    aten::AssetManager::registerMtrl(
        "sp_00_vrata_kock",
        createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.784314, 0.784314, 0.784314), VRATA_KO, nullptr));

    std::vector<aten::object*> objs;

    aten::ObjLoader::load(objs, "../../asset/sponza/sponza.obj", ctxt);

    objs[0]->importInternalAccelTree("../../asset/sponza/sponza.sbvh", ctxt, 0);

    auto sponza = aten::TransformableFactory::createInstance<aten::object>(ctxt, objs[0], aten::mat4::Identity);

#if 1
    {
        int offsetTriIdx = ctxt.getTriangleNum();

        objs.clear();
        aten::ObjLoader::load(objs, "../../asset/sponza/sponza_lod.obj", ctxt);
        objs[0]->importInternalAccelTree("../../asset/sponza/sponza_lod.sbvh", ctxt, offsetTriIdx);
        sponza->setLod(objs[0]);
    }
#endif

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

/////////////////////////////////////////////////////

void BunnyScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    aten::MaterialParameter mtrlParam;
    mtrlParam.baseColor = aten::vec3(0.7, 0.7, 0.7);
    mtrlParam.ior = 1.3;

    aten::AssetManager::registerMtrl(
        "m1",
        createMaterialWithParamter(ctxt, aten::MaterialType::Lambert_Refraction, mtrlParam));

    std::vector<aten::object*> objs;

    aten::ObjLoader::load(objs, "../../asset/teapot/teapot.obj", ctxt);
    auto bunny = aten::TransformableFactory::createInstance<aten::object>(ctxt, objs[0], aten::mat4::Identity);
    scene->add(bunny);
}

void BunnyScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    real& fov)
{
    pos = aten::vec3(0.f, 1.f, 100.f);
    at = aten::vec3(0.f, 1.f, 0.f);
    fov = 45;
}

/////////////////////////////////////////////////////

void DeformScene::makeScene(
    aten::context& ctxt,
    aten::scene* scene)
{
    aten::deformable* mdl = aten::TransformableFactory::createDeformable(ctxt);
    mdl->read("unitychan_gpu.mdl");

    aten::ImageLoader::setBasePath("../../asset/unitychan/Texture");
    aten::MaterialLoader::load("unitychan_mtrl.xml", ctxt);

    auto deformMdl = aten::TransformableFactory::createInstance<aten::deformable>(ctxt, mdl, aten::mat4::Identity);
    scene->add(deformMdl);

    s_deformMdl = deformMdl;

    aten::ImageLoader::setBasePath("./");

    s_deformAnm = new aten::DeformAnimation();
    s_deformAnm->read("unitychan.anm");
}

void DeformScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    real& fov)
{
    pos = aten::vec3(0.f, 1.f, 3.f);
    at = aten::vec3(0.f, 1.f, 0.f);
    fov = 45;
}

/////////////////////////////////////////////////////

void DeformInBoxScene::makeScene(
    aten::context& ctxt,
    aten::scene* scene)
{
#if 1
    {
        auto emit = createMaterial(ctxt, aten::MaterialType::Emissive, aten::vec3(36, 33, 24));
        aten::AssetManager::registerMtrl(
            "light",
            emit);

        aten::AssetManager::registerMtrl(
            "backWall",
            createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.580000, 0.568000, 0.544000)));
        aten::AssetManager::registerMtrl(
            "ceiling",
            createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.580000, 0.568000, 0.544000)));
        aten::AssetManager::registerMtrl(
            "floor",
            createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.580000, 0.568000, 0.544000)));
        aten::AssetManager::registerMtrl(
            "leftWall",
            createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.504000, 0.052000, 0.040000)));

        aten::AssetManager::registerMtrl(
            "rightWall",
            createMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.112000, 0.360000, 0.072800)));

        std::vector<aten::object*> objs;

        aten::ObjLoader::load(objs, "../../asset/cornellbox/box.obj", ctxt, false);

        auto light = aten::TransformableFactory::createInstance<aten::object>(
            ctxt,
            objs[0],
            aten::vec3(0),
            aten::vec3(0),
            aten::vec3(1));
        scene->add(light);

        auto areaLight = new aten::AreaLight(light, emit->param().baseColor);
        scene->addLight(areaLight);

        auto box = aten::TransformableFactory::createInstance<aten::object>(ctxt, objs[1], aten::mat4::Identity);
        scene->add(box);
    }
#endif

    {
        aten::deformable* mdl = aten::TransformableFactory::createDeformable(ctxt);
        mdl->read("unitychan_gpu.mdl");

        aten::ImageLoader::setBasePath("../../asset/unitychan/Texture");
        aten::MaterialLoader::load("unitychan_mtrl.xml", ctxt);

        auto deformMdl = aten::TransformableFactory::createInstance<aten::deformable>(ctxt, mdl, aten::mat4::Identity);
        scene->add(deformMdl);

        s_deformMdl = deformMdl;

        aten::ImageLoader::setBasePath("./");

        s_deformAnm = new aten::DeformAnimation();
        s_deformAnm->read("unitychan.anm");
    }
}

void DeformInBoxScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    real& fov)
{
    pos = aten::vec3(0.f, 1.f, 3.f);
    at = aten::vec3(0.f, 1.f, 0.f);
    fov = 45;
}
