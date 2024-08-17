#include "scenedefs.h"

#include <array>

static std::shared_ptr<aten::material> CreateMaterial(
    std::string_view name,
    aten::context& ctxt,
    aten::MaterialType type,
    const aten::vec3& albedo,
    aten::texture* albedoMap,
    aten::texture* normalMap)
{
    aten::MaterialParameter param;
    param.type = type;
    param.baseColor = albedo;

    auto mtrl = ctxt.CreateMaterialWithMaterialParameter(
        name,
        param,
        albedoMap,
        normalMap,
        nullptr);

    return mtrl;
}

static std::shared_ptr<aten::material> CreateMaterial(
    std::string_view name,
    aten::context& ctxt,
    aten::MaterialType type,
    const aten::vec3& albedo)
{
    aten::MaterialParameter param;
    param.type = type;
    param.baseColor = albedo;

    return ctxt.CreateMaterialWithMaterialParameter(
        name,
        param,
        nullptr, nullptr, nullptr);
}

static std::shared_ptr<aten::material> createMaterialWithParamter(
    std::string_view name,
    aten::context& ctxt,
    aten::MaterialType type,
    const aten::MaterialParameter& param)
{
    aten::MaterialParameter mtrl_param = param;
    mtrl_param.type = type;

    return ctxt.CreateMaterialWithMaterialParameter(
        name,
        mtrl_param,
        nullptr, nullptr, nullptr);
}

static std::shared_ptr<aten::material> createMaterialWithParamter(
    std::string_view name,
    aten::context& ctxt,
    aten::MaterialType type,
    const aten::MaterialParameter& param,
    aten::texture* albedoMap,
    aten::texture* normalMap,
    aten::texture* roughnessMap)
{
    aten::MaterialParameter mtrl_param = param;
    mtrl_param.type = type;

    return ctxt.CreateMaterialWithMaterialParameter(
        name,
        mtrl_param,
        albedoMap,
        normalMap,
        roughnessMap);
}

void CornellBoxScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    auto emit = CreateMaterial("emit", ctxt, aten::MaterialType::Emissive, aten::vec3(1.0f, 1.0f, 1.0f));

    auto light = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(50.0, 75.0, 81.6),
        5.0,
        emit);

    constexpr auto r = 1e3f;

    auto left = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(r + 1.0f, 40.8f, 81.6f),
        r,
        CreateMaterial("left", ctxt, aten::MaterialType::Lambert, aten::vec3(0.75f, 0.25f, 0.25f)));

    auto right = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(-r + 99.0f, 40.8f, 81.6f),
        r,
        CreateMaterial("right", ctxt, aten::MaterialType::Lambert, aten::vec3(0.25f, 0.25f, 0.75f)));

    auto wall = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(50.0f, 40.8f, r),
        r,
        CreateMaterial("wall", ctxt, aten::MaterialType::Lambert, aten::vec3(0.75f, 0.75f, 0.75f)));

    auto floor = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(50.0f, r, 81.6f),
        r,
        CreateMaterial("floor", ctxt, aten::MaterialType::Lambert, aten::vec3(0.75f, 0.75f, 0.75f)));

    auto ceil = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(50.0f, -r + 81.6f, 81.6f),
        r,
        CreateMaterial("ceil", ctxt, aten::MaterialType::Lambert, aten::vec3(0.75f, 0.75f, 0.75f)));

    //auto tex = aten::ImageLoader::load("../../asset/earth.bmp");

#define DEFALT    (1)

#if DEFALT
    auto green = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(65.0f, 20.0f, 20.0f),
        20.0f,
        CreateMaterial("green", ctxt, aten::MaterialType::Lambert, aten::vec3(0.25f, 0.75f, 0.25f)));
        //CreateMaterial(ctxt, aten::vec3(1, 1, 1), tex));
#else
    auto green = aten::TransformableFactory::createSphere(
        aten::vec3(65, 20, 20),
        20,
        new aten::MicrofacetGGX(aten::vec3(0.7, 0.6, 0.5), 0.2, 0.2));
#endif

#if DEFALT
    auto mirror = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(27, 16.5, 47),
        16.5,
        CreateMaterial("mirror", ctxt, aten::MaterialType::Specular, aten::vec3(0.99, 0.99, 0.99)));
#else
    auto spec = new aten::MicrofacetBlinn(aten::vec3(1, 1, 1), 200, 0.8);
    auto diff = CreateMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(0.0, 0.7, 0.0));

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
    CreateMaterial("m1", ctxt, aten::MaterialType::Lambert, aten::vec3(0.2, 0.2, 0.7));
    CreateMaterial("Material.001", ctxt, aten::MaterialType::Lambert, aten::vec3(0.2, 0.2, 0.7));

    auto obj = aten::ObjLoader::LoadFirstObj("../../asset/suzanne/suzanne.obj", ctxt);
    //auto obj = aten::ObjLoader::load("../../asset/teapot.obj");

    aten::mat4 mtx_L2W;
    mtx_L2W.asRotateByY(aten::Deg2Rad(-25));

    aten::mat4 mtxT;
    mtxT.asTrans(aten::vec3(77, 16.5, 78));

    aten::mat4 mtxS;
    mtxS.asScale(10);

    mtx_L2W = mtxT * mtx_L2W * mtxS;

    auto glass = aten::TransformableFactory::createInstance<aten::PolygonObject>(ctxt, obj, mtx_L2W);
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
    auto l = std::make_shared<aten::AreaLight>(light, emit->param().baseColor, 400.0f);
#else
    auto l = std::make_shared<aten::AreaLight>(glass, emit->param().baseColor, 400.0f);
#endif

    ctxt.AddLight(l);
}

void CornellBoxScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
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
        aten::vec3(0.0f, -1000.0f, 0.0f),
        1000.0f,
        CreateMaterial("floor", ctxt, aten::MaterialType::Lambert, aten::vec3(0.8f, 0.8f, 0.8f)));
    scene->add(s);

    aten::MaterialParameter mtrlParam;

    int32_t i = 1;
    for (int32_t x = -11; x < 11; x++) {
        for (int32_t z = -11; z < 11; z++) {
            auto choose_mtrl = aten::drand48();

            aten::vec3 center = aten::vec3(
                x + 0.9f * aten::drand48(),
                0.2f,
                z + 0.9f * aten::drand48());

            const auto mtrl_name = aten::StringFormat("%d", i);

            if (length(center - aten::vec3(4.0f, 0.2f, 0.0f)) > 0.9f) {
                if (choose_mtrl < 0.8f) {
                    // lambert
                    s = aten::TransformableFactory::createSphere(
                        ctxt,
                        center,
                        0.2f,
                        CreateMaterial(mtrl_name, ctxt, aten::MaterialType::Lambert, aten::vec3(aten::drand48() * aten::drand48(), aten::drand48() * aten::drand48(), aten::drand48() * aten::drand48())));
                }
                else if (choose_mtrl < 0.95) {
                    // specular
                    s = aten::TransformableFactory::createSphere(
                        ctxt,
                        center,
                        0.2f,
                        CreateMaterial(mtrl_name, ctxt, aten::MaterialType::Specular, aten::vec3(0.5 * (1 + aten::drand48()), 0.5 * (1 + aten::drand48()), 0.5 * (1 + aten::drand48()))));
                }
                else {
                    // glass
                    mtrlParam.baseColor = aten::vec3(1);
                    mtrlParam.standard.ior = 1.5;

                    s = aten::TransformableFactory::createSphere(
                        ctxt,
                        center,
                        0.2f,
                        createMaterialWithParamter(mtrl_name, ctxt, aten::MaterialType::Refraction, mtrlParam));
                }

                scene->add(s);
            }
        }
    }

    mtrlParam.baseColor = aten::vec3(1);
    mtrlParam.standard.ior = 1.5;
    s = aten::TransformableFactory::createSphere(ctxt, aten::vec3(0, 1, 0), 1.0, createMaterialWithParamter("refraction", ctxt, aten::MaterialType::Refraction, mtrlParam));
    scene->add(s);

    s = aten::TransformableFactory::createSphere(ctxt, aten::vec3(-4, 1, 0), 1.0, CreateMaterial("lambert", ctxt, aten::MaterialType::Lambert, aten::vec3(0.4, 0.2, 0.1)));
    scene->add(s);

    s = aten::TransformableFactory::createSphere(ctxt, aten::vec3(4, 1, 0), 1.0, CreateMaterial("specular", ctxt, aten::MaterialType::Specular, aten::vec3(0.7, 0.6, 0.5)));
    scene->add(s);
}

void RandomScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
    pos = aten::vec3(13, 2, 3);
    at = aten::vec3(0, 0, 0);
    fov = 30;
}

/////////////////////////////////////////////////////

void MtrlTestScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    aten::MaterialParameter mtrlParam;

    mtrlParam.baseColor = aten::vec3(0.7f, 0.6f, 0.5f);
    mtrlParam.standard.roughness = 0.2f;
    mtrlParam.standard.ior = 0.2f;
    auto s_ggx = aten::TransformableFactory::createSphere(ctxt, aten::vec3(-3.0f, 0.0f, 0.0f), 1.0f, createMaterialWithParamter("ggx", ctxt, aten::MaterialType::GGX, mtrlParam));
    scene->add(s_ggx);

    mtrlParam.baseColor = aten::vec3(0.7f, 0.6f, 0.5f);
    mtrlParam.standard.roughness = 0.2f;
    mtrlParam.standard.ior = 0.2f;
    auto s_beckman = aten::TransformableFactory::createSphere(ctxt, aten::vec3(+1.0f, 0.0f, 0.0f), 1.0f, createMaterialWithParamter("beckman", ctxt, aten::MaterialType::Beckman, mtrlParam));
    scene->add(s_beckman);

    auto s_glass = aten::TransformableFactory::createSphere(ctxt, aten::vec3(+3.0f, 0.0f, 0.0f), 1.0f, CreateMaterial("specular", ctxt, aten::MaterialType::Specular, aten::vec3(0.7, 0.6, 0.5)));
    scene->add(s_glass);
}

void MtrlTestScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
    pos = aten::vec3(0.0f, 0.0f, 13.0f);
    at = aten::vec3(0.0f, 0.0f, 0.0f);
    fov = 30.0f;
}

/////////////////////////////////////////////////////

void ObjectScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    aten::MaterialParameter mtrlParam;
    mtrlParam.baseColor = aten::vec3(0.7f, 0.6f, 0.5f);
    mtrlParam.standard.roughness = 0.5f;
    mtrlParam.standard.ior = 0.2f;

    createMaterialWithParamter("m1", ctxt, aten::MaterialType::GGX, mtrlParam);
    createMaterialWithParamter("Material.001", ctxt, aten::MaterialType::GGX, mtrlParam);

    auto obj = aten::ObjLoader::LoadFirstObj("../../asset/suzanne/suzanne.obj", ctxt);
    //auto obj = aten::ObjLoader::load("../../asset/teapot.obj");

    aten::mat4 mtx_L2W;
    mtx_L2W.asRotateByZ(aten::Deg2Rad(45.0f));

    aten::mat4 mm;
    mm.asTrans(aten::vec3(-1.0f, 0.0f, 0.0f));

    mtx_L2W = mtx_L2W * mm;

    auto instance = aten::TransformableFactory::createInstance<aten::PolygonObject>(ctxt, obj, mtx_L2W);

    scene->add(instance);
}

void ObjectScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
    pos = aten::vec3(0.0f, 0.0f, 10.0f);
    //pos = aten::vec3(0.0, 0.0, 60.0);
    at = aten::vec3(0.0f, 0.0f, 0.0f);
    fov = 30.0f;
}

/////////////////////////////////////////////////////

void PointLightScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    constexpr auto r = 1e3f;

    auto floor = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(0.0f, -r, 0.0f),
        r,
        CreateMaterial("floor", ctxt, aten::MaterialType::Lambert, aten::vec3(0.75f, 0.75f, 0.75f)));

    auto green = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(0.65f, 0.2f, 0.2f),
        0.2f,
        CreateMaterial("green", ctxt, aten::MaterialType::Lambert, aten::vec3(0.25f, 0.75f, 0.25f)));

    scene->add(floor);
    scene->add(green);

    auto l = std::make_shared<aten::PointLight>(
        aten::vec3(0.5f, 1.9f, 0.816f),
        aten::vec3(1.0f, 1.0f, 1.0f),
        100.0f);

    ctxt.AddLight(l);
}

void PointLightScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
    pos = aten::vec3(0.5f, 0.52f, 2.95f);
    at = aten::vec3(0.5f, 0.408f, 1.19f);
    fov = 30.0f;
}

/////////////////////////////////////////////////////

void DirectionalLightScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    constexpr auto r = 1e5f;

    auto floor = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(0.0f, -r, 0.0f),
        r,
        CreateMaterial("floor", ctxt, aten::MaterialType::Lambert, aten::vec3(0.75f, 0.75f, 0.75f)));

    auto green = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(65.0f, 20.0f, 20.0f),
        20,
        CreateMaterial("green", ctxt, aten::MaterialType::Lambert, aten::vec3(0.25f, 0.75f, 0.25f)));

    scene->add(floor);
    scene->add(green);

    auto l = std::make_shared<aten::DirectionalLight>(
        aten::vec3(1.0f, -1.0f, 1.0f),
        aten::vec3(1.0f, 1.0f, 1.0f),
        100.0f);

    ctxt.AddLight(l);
}

void DirectionalLightScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
    pos = aten::vec3(50.0f, 52.0f, 295.6f);
    at = aten::vec3(50.0f, 40.8f, 119.0f);
    fov = 30.0f;
}

/////////////////////////////////////////////////////

void SpotLightScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    constexpr auto r = 1e3f;

    auto floor = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(0.0f, -r, 0.0f),
        r,
        CreateMaterial("floor", ctxt, aten::MaterialType::Lambert, aten::vec3(0.75f, 0.75f, 0.75f)));

    auto green = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(0.65f, 0.2f, 0.2f),
        0.2f,
        CreateMaterial("green", ctxt, aten::MaterialType::Lambert, aten::vec3(0.25f, 0.75f, 0.25f)));

    auto red = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(0.25f, 0.2f, 0.2f),
        0.2f,
        CreateMaterial("red", ctxt, aten::MaterialType::Lambert, aten::vec3(0.75f, 0.25f, 0.25f)));

    scene->add(floor);
    scene->add(green);
    scene->add(red);

    auto l = std::make_shared<aten::SpotLight>(
        aten::vec3(0.65, 1.9f, 0.2f),
        aten::vec3(0, -1, 0),
        aten::vec3(1.0f, 1.0f, 1.0f),
        400.0f,
        float(aten::Deg2Rad(30)),
        float(aten::Deg2Rad(60)));

    ctxt.AddLight(l);
}

void SpotLightScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
    pos = aten::vec3(0.5f, 0.52f, 2.956f);
    at = aten::vec3(0.5f, 0.408f, 1.19f);
    fov = 30.0f;
}

/////////////////////////////////////////////////////

void ManyLightScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    auto s = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(0, -1000, 0),
        1000,
        CreateMaterial("floor", ctxt, aten::MaterialType::Lambert, aten::vec3(0.8, 0.8, 0.8)));
    scene->add(s);

    aten::MaterialParameter mtrlParam;

#if 1
    int32_t i = 1;
    for (int32_t x = -5; x < 5; x++) {
        for (int32_t z = -5; z < 5; z++) {
            auto choose_mtrl = aten::drand48();

            aten::vec3 center = aten::vec3(
                x + 0.9f * aten::drand48(),
                0.2f,
                z + 0.9f * aten::drand48());

            const auto mtrl_name = aten::StringFormat("%d", i);

            if (length(center - aten::vec3(4.0f, 0.2f, 0.0f)) > 0.9f) {
                if (choose_mtrl < 0.8f) {
                    // lambert
                    s = aten::TransformableFactory::createSphere(
                        ctxt,
                        center,
                        0.2f,
                        CreateMaterial(mtrl_name, ctxt, aten::MaterialType::Lambert, aten::vec3(aten::drand48() * aten::drand48(), aten::drand48() * aten::drand48(), aten::drand48() * aten::drand48())));
                }
                else if (choose_mtrl < 0.95f) {
                    // specular
                    s = aten::TransformableFactory::createSphere(
                        ctxt,
                        center,
                        0.2f,
                        CreateMaterial(mtrl_name, ctxt, aten::MaterialType::Specular, aten::vec3(0.5f * (1.0f + aten::drand48()), 0.5f * (1.0f + aten::drand48()), 0.5f * (1.0f + aten::drand48()))));
                }
                else {
                    // glass
                    mtrlParam.baseColor = aten::vec3(1.0f);
                    mtrlParam.standard.ior = 1.5f;

                    s = aten::TransformableFactory::createSphere(ctxt, center, 0.2f, createMaterialWithParamter(mtrl_name, ctxt, aten::MaterialType::Refraction, mtrlParam));
                }

                scene->add(s);
            }
        }
    }
#endif

    mtrlParam.baseColor = aten::vec3(1);
    mtrlParam.standard.ior = 1.5;
    s = aten::TransformableFactory::createSphere(ctxt, aten::vec3(0, 1, 0), 1.0, createMaterialWithParamter("refraction", ctxt, aten::MaterialType::Refraction, mtrlParam));
    scene->add(s);

    s = aten::TransformableFactory::createSphere(ctxt, aten::vec3(-4, 1, 0), 1.0, CreateMaterial("lambert", ctxt, aten::MaterialType::Lambert, aten::vec3(0.8, 0.2, 0.1)));
    scene->add(s);

    s = aten::TransformableFactory::createSphere(ctxt, aten::vec3(4, 1, 0), 1.0, CreateMaterial("specular", ctxt, aten::MaterialType::Specular, aten::vec3(0.7, 0.6, 0.5)));
    scene->add(s);

    auto dir = std::make_shared<aten::DirectionalLight>(aten::vec3(-1, -1, -1), aten::vec3(0.5, 0.5, 0.5), 400.0f);
    auto point = std::make_shared<aten::PointLight>(
        aten::vec3(0, 10, -1),
        aten::vec3(0.0f, 0.0f, 1.0f),
        400.0f);
    auto spot = std::make_shared<aten::SpotLight>(
        aten::vec3(0, 5, 0),
        aten::vec3(0, -1, 0),
        aten::vec3(0.0f, 1.0f, 0.0f),
        400.0f,
        float(aten::Deg2Rad(30)),
        float(aten::Deg2Rad(60)));

    ctxt.AddLight(dir);
    ctxt.AddLight(spot);
    ctxt.AddLight(point);
}

void ManyLightScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
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
    mtrlParam.standard.roughness = 0.5f;
    mtrlParam.standard.ior = 0.2f;

    auto blinn = createMaterialWithParamter(
        "ggx",
        ctxt, aten::MaterialType::GGX, mtrlParam,
        albedo.get(), nml.get(), nullptr);

    auto s_blinn = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(-3.0f, 0.0f, 0.0f),
        1.0f,
        blinn);
    scene->add(s_blinn);

#if 1
    mtrlParam.baseColor = clr;
    mtrlParam.standard.roughness = 0.2f;
    mtrlParam.standard.ior = 0.2f;

    auto ggx = createMaterialWithParamter(
        "ggx_1",
        ctxt, aten::MaterialType::GGX, mtrlParam,
        albedo.get(), nml.get(), rough.get());

    auto s_ggx = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(-1.0f, 0.0f, 0.0f),
        1.0f,
        ggx);
    scene->add(s_ggx);

    mtrlParam.baseColor = clr;
    mtrlParam.standard.roughness = 0.2f;
    mtrlParam.standard.ior = 0.2f;

    auto beckman = createMaterialWithParamter(
        "beckman",
        ctxt, aten::MaterialType::Beckman, mtrlParam,
        albedo.get(), nml.get(), rough.get());

    auto s_beckman = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(+1.0f, 0.0f, 0.0f),
        1.0f,
        beckman);
    scene->add(s_beckman);

    mtrlParam.baseColor = clr;
    mtrlParam.standard.roughness = 0.2f;
    mtrlParam.standard.ior = 0.2f;

    auto lambert = CreateMaterial(
        "lambert",
        ctxt, aten::MaterialType::Lambert, clr,
        albedo.get(), nml.get());

    auto s_lambert = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(+3.0f, 0.0f, 0.0f),
        1.0f,
        lambert);
    scene->add(s_lambert);

    auto specular = CreateMaterial(
        "specular",
        ctxt, aten::MaterialType::Specular, clr,
        nullptr, nml_2.get());

    auto s_spec = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(-3.0f, +2.0f, 0.0f),
        1.0f,
        specular);
    scene->add(s_spec);

    auto s_ref = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(-1.0f, +2.0f, 0.0f),
        1.0f,
        specular);
    scene->add(s_ref);
#endif
}

void TexturesScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
    pos = aten::vec3(0.0f, 0.0f, 13.0f);
    at = aten::vec3(0.0f, 0.0f, 0.0f);
    fov = 30.0f;
}

/////////////////////////////////////////////////////

void HideLightScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    auto emit = CreateMaterial("emit", ctxt, aten::MaterialType::Emissive, aten::vec3(1.0f, 1.0f, 1.0f));

    auto light = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(50.0f, 90.0f, 81.6f),
        15.0f,
        emit);

    constexpr auto r = 1e3f;

    auto left = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(r + 1.0f, 40.8f, 81.6f),
        r,
        CreateMaterial("left", ctxt, aten::MaterialType::Lambert, aten::vec3(0.75f, 0.25f, 0.25f)));

    auto right = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(-r + 99.0f, 40.8f, 81.6f),
        r,
        CreateMaterial("right", ctxt, aten::MaterialType::Lambert, aten::vec3(0.25f, 0.25f, 0.75f)));

    auto wall = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(50.0f, 40.8f, r),
        r,
        CreateMaterial("wall", ctxt, aten::MaterialType::Lambert, aten::vec3(0.75f, 0.75f, 0.75f)));

    auto floor = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(50.0f, r, 81.6f),
        r,
        CreateMaterial("floor", ctxt, aten::MaterialType::Lambert, aten::vec3(0.75f, 0.75f, 0.75f)));

    auto ceil = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(50, -r + 81.6, 81.6),
        r,
        CreateMaterial("ceil", ctxt, aten::MaterialType::Lambert, aten::vec3(0.75f, 0.75f, 0.75f)));

    //auto tex = aten::ImageLoader::load("../../asset/earth.bmp");

    auto green = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(65.0f, 20.0f, 20.0f),
        20.0f,
        CreateMaterial("green", ctxt, aten::MaterialType::Lambert, aten::vec3(0.25f, 0.75f, 0.25f)));
    //CreateMaterial(ctxt, aten::MaterialType::Lambert, aten::vec3(1, 1, 1), tex));

    auto mirror = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(27, 16.5, 47),
        16.5,
        CreateMaterial("mirror", ctxt, aten::MaterialType::Specular, aten::vec3(0.99, 0.99, 0.99)));

    aten::MaterialParameter mtrlParam;
    mtrlParam.baseColor = aten::vec3(0.99f, 0.99f, 0.99f);
    mtrlParam.standard.ior = 1.5;
    auto glass = aten::TransformableFactory::createSphere(
        ctxt,
        aten::vec3(77.0f, 16.5f, 78.0f),
        16.5f,
        createMaterialWithParamter("glass", ctxt, aten::MaterialType::Refraction, mtrlParam));

#if 1
    scene->add(light);
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

    auto l = std::make_shared<aten::AreaLight>(light, emit->param().baseColor, 400.0f);

    ctxt.AddLight(l);
#endif
}

void HideLightScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
    pos = aten::vec3(50.0f, 52.0f, 295.6f);
    at = aten::vec3(50.0f, 40.8f, 119.0f);
    fov = 30.0f;
}

/////////////////////////////////////////////////////

std::shared_ptr<aten::instance<aten::PolygonObject>> ObjCornellBoxScene::makeScene(
    aten::context& ctxt, aten::scene* scene)
{
    auto emit = CreateMaterial("light", ctxt, aten::MaterialType::Emissive, aten::vec3(1.0f, 1.0f, 1.0f));

    auto objs = aten::ObjLoader::load("../../asset/cornellbox/orig.obj", ctxt,
        [&](std::string_view name, aten::context& ctxt,
            aten::MaterialType type, const aten::vec3& mtrl_clr,
            const std::string& albedo, const std::string& nml) -> auto {
                (void)albedo;
                (void)nml;

                if (name == "shortBox") {
                    //type = aten::MaterialType::GGX;
                    type = aten::MaterialType::Specular;

                    aten::MaterialParameter mtrlParam;
                    mtrlParam.baseColor = aten::vec3(0.7f, 0.6f, 0.5f);
                    mtrlParam.standard.roughness = 0.1f;
                    mtrlParam.standard.ior = 0.01f;

                    auto mtrl = createMaterialWithParamter(name, ctxt, type, mtrlParam);
                    return mtrl;
                }
                else if (name == "floor") {
                    type = aten::MaterialType::GGX;
                    //type = aten::MaterialType::Specular;

                    aten::MaterialParameter mtrlParam;
                    mtrlParam.baseColor = aten::vec3(0.7f, 0.6f, 0.5f);
                    mtrlParam.standard.roughness = 0.1f;
                    mtrlParam.standard.ior = 0.01f;

                    auto mtrl = createMaterialWithParamter(name, ctxt, type, mtrlParam);
                    return mtrl;
                }
                else {
                    auto mtrl = CreateMaterial(name, ctxt, type, mtrl_clr);
                    return mtrl;
                }
        },
        true, true);

    auto light = aten::TransformableFactory::createInstance<aten::PolygonObject>(
        ctxt,
        objs[0],
        aten::vec3(0.0f),
        aten::vec3(0.0f),
        aten::vec3(1.0f));
    scene->add(light);

    auto areaLight = std::make_shared<aten::AreaLight>(light, emit->param().baseColor, 200.0f);
    ctxt.AddLight(areaLight);

    for (int32_t i = 1; i < objs.size(); i++) {
        auto box = aten::TransformableFactory::createInstance<aten::PolygonObject>(ctxt, objs[i], aten::mat4::Identity);
        scene->add(box);
    }

    return light;
}

void ObjCornellBoxScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
    pos = aten::vec3(0.f, 1.f, 3.f);
    at = aten::vec3(0.f, 1.f, 0.f);
    fov = 45.0f;
}

/////////////////////////////////////////////////////

void SponzaScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    auto objs = aten::ObjLoader::load(
        "../../asset/sponza/sponza.obj", ctxt,
        [&](std::string_view name, aten::context& ctxt,
            aten::MaterialType type, const aten::vec3& mtrl_clr,
            const std::string& albedo, const std::string& nml) -> auto {
                auto albedo_map = albedo.empty()
                    ? nullptr
                    : aten::ImageLoader::load("../../asset/sponza/" + albedo, ctxt);
                auto nml_map = nml.empty()
                    ? nullptr
                    : aten::ImageLoader::load("../../asset/sponza/" + nml, ctxt);

                auto mtrl = CreateMaterial(name, ctxt, type, mtrl_clr, albedo_map.get(), nml_map.get());
                return mtrl;
        });

    objs[0]->importInternalAccelTree("../../asset/sponza/sponza.sbvh", ctxt, 0);

    auto sponza = aten::TransformableFactory::createInstance<aten::PolygonObject>(ctxt, objs[0], aten::mat4::Identity);

#if 1
    {
        int32_t offsetTriIdx = ctxt.GetTriangleNum();

        objs.clear();
        objs = aten::ObjLoader::load("../../asset/sponza/sponza_lod.obj", ctxt);
        objs[0]->importInternalAccelTree("../../asset/sponza/sponza_lod.sbvh", ctxt, offsetTriIdx);
        sponza->setLod(objs[0]);
    }
#endif

    scene->add(sponza);
}

void SponzaScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
#if 1
    pos = aten::vec3(0.f, 1.f, 3.f);
    at = aten::vec3(0.f, 1.f, 0.f);
#else
    pos = aten::vec3(-3.09f, 3.40f, -0.13f);
    at = aten::vec3(-2.09f, 3.33f, -0.13f);
#endif
    fov = 45.0f;
}

/////////////////////////////////////////////////////

void BunnyScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    aten::MaterialParameter mtrlParam;
    mtrlParam.baseColor = aten::vec3(0.7f, 0.7f, 0.7f);

    createMaterialWithParamter("m1", ctxt, aten::MaterialType::Lambert, mtrlParam);

    auto objs = aten::ObjLoader::load("../../asset/teapot/teapot.obj", ctxt);
    auto bunny = aten::TransformableFactory::createInstance<aten::PolygonObject>(ctxt, objs[0], aten::mat4::Identity);
    scene->add(bunny);
}

void BunnyScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
    pos = aten::vec3(0.f, 1.f, 100.f);
    at = aten::vec3(0.f, 1.f, 0.f);
    fov = 45.0f;
}

/////////////////////////////////////////////////////

aten::tuple<std::shared_ptr<aten::instance<aten::deformable>>, std::shared_ptr<aten::DeformAnimation>> DeformScene::makeScene(
    aten::context& ctxt, aten::scene* scene)
{
    auto mdl = aten::TransformableFactory::createDeformable(ctxt);
    mdl->read("../../asset/converted_unitychan/unitychan_gpu.mdl");

    aten::ImageLoader::setBasePath("../../asset/unitychan/Texture");
    aten::MaterialLoader::load("../../asset/converted_unitychan/unitychan_mtrl.xml", ctxt);

    auto deformMdl = aten::TransformableFactory::createInstance<aten::deformable>(ctxt, mdl, aten::mat4::Identity);
    scene->add(deformMdl);

    aten::ImageLoader::setBasePath("./");

    auto deformAnm = std::make_shared<aten::DeformAnimation>();
    deformAnm->read("../../asset/converted_unitychan/unitychan_WAIT00.anm");

    return aten::make_tuple(deformMdl, deformAnm);
}

void DeformScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
    pos = aten::vec3(0.f, 1.f, 3.f);
    at = aten::vec3(0.f, 1.f, 0.f);
    fov = 45.0f;
}

/////////////////////////////////////////////////////

aten::tuple<std::shared_ptr<aten::instance<aten::deformable>>, std::shared_ptr<aten::DeformAnimation>> DeformInBoxScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
#if 1
    {
        auto emit = CreateMaterial("light", ctxt, aten::MaterialType::Emissive, aten::vec3(1.0f, 1.0f, 1.0f));

        CreateMaterial("backWall", ctxt, aten::MaterialType::Lambert, aten::vec3(0.580000f, 0.568000f, 0.544000f));
        CreateMaterial("ceiling", ctxt, aten::MaterialType::Lambert, aten::vec3(0.580000f, 0.568000f, 0.544000f));
        CreateMaterial("floor", ctxt, aten::MaterialType::Lambert, aten::vec3(0.580000f, 0.568000f, 0.544000f));
        CreateMaterial("leftWall", ctxt, aten::MaterialType::Lambert, aten::vec3(0.504000f, 0.052000f, 0.040000f));
        CreateMaterial("rightWall", ctxt, aten::MaterialType::Lambert, aten::vec3(0.112000f, 0.360000f, 0.072800f));

        auto objs = aten::ObjLoader::load("../../asset/cornellbox/box.obj", ctxt, nullptr, false);

        auto light = aten::TransformableFactory::createInstance<aten::PolygonObject>(
            ctxt,
            objs[0],
            aten::vec3(0.0f),
            aten::vec3(0.0f),
            aten::vec3(1.0f));
        scene->add(light);

        auto areaLight = std::make_shared<aten::AreaLight>(light, emit->param().baseColor, 400.0f);
        ctxt.AddLight(areaLight);

        auto box = aten::TransformableFactory::createInstance<aten::PolygonObject>(ctxt, objs[1], aten::mat4::Identity);
        scene->add(box);
    }
#endif

    auto mdl = aten::TransformableFactory::createDeformable(ctxt);
    mdl->read("../../asset/converted_unitychan/unitychan_gpu.mdl");

    aten::ImageLoader::setBasePath("../../asset/unitychan/Texture");
    aten::MaterialLoader::load("../../asset/converted_unitychan/unitychan_mtrl.xml", ctxt);

    auto deformMdl = aten::TransformableFactory::createInstance<aten::deformable>(ctxt, mdl, aten::mat4::Identity);
    scene->add(deformMdl);

    aten::ImageLoader::setBasePath("./");

    auto deformAnm = std::make_shared<aten::DeformAnimation>();
    deformAnm->read("../../asset/converted_unitychan/unitychan_WAIT00.anm");

    return aten::make_tuple(deformMdl, deformAnm);
}

void DeformInBoxScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
    pos = aten::vec3(0.f, 1.f, 3.f);
    at = aten::vec3(0.f, 1.f, 0.f);
    fov = 45;
}

/////////////////////////////////////////////////////

void AlphaBlendedObjCornellBoxScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    auto back = CreateMaterial("backWall", ctxt, aten::MaterialType::Lambert, aten::vec3(0.580000f, 0.568000f, 0.544000f));
    back->param().baseColor.a = 0.0f;

    CreateMaterial("ceiling", ctxt, aten::MaterialType::Lambert, aten::vec3(0.580000f, 0.568000f, 0.544000f));
    CreateMaterial("floor", ctxt, aten::MaterialType::Lambert, aten::vec3(0.580000f, 0.568000f, 0.544000f));

    auto left = CreateMaterial("leftWall", ctxt, aten::MaterialType::Lambert, aten::vec3(0.504000f, 0.052000f, 0.040000f));
    left->param().baseColor.a = 0.5f;

    auto emit = CreateMaterial("light", ctxt, aten::MaterialType::Emissive, aten::vec3(1.0f, 1.0f, 1.0f));

    CreateMaterial("rightWall", ctxt, aten::MaterialType::Lambert, aten::vec3(0.112000f, 0.360000f, 0.072800f));
    CreateMaterial("shortBox", ctxt, aten::MaterialType::Lambert, aten::vec3(0.580000f, 0.568000f, 0.544000f));

    auto tall = CreateMaterial("tallBox", ctxt, aten::MaterialType::Lambert, aten::vec3(0.0000f, 0.000f, 1.0000f));
    tall->param().baseColor.a = 0.25f;

    auto objs = aten::ObjLoader::load("../../asset/cornellbox/orig.obj", ctxt, nullptr, true, true);

    auto light = aten::TransformableFactory::createInstance<aten::PolygonObject>(
        ctxt,
        objs[0],
        aten::vec3(0.0f),
        aten::vec3(0.0f),
        aten::vec3(1.0f));
    scene->add(light);

    auto areaLight = std::make_shared<aten::AreaLight>(light, emit->param().baseColor, 400.0f);
    ctxt.AddLight(areaLight);

    for (int32_t i = 1; i < objs.size(); i++) {
        auto box = aten::TransformableFactory::createInstance<aten::PolygonObject>(ctxt, objs[i], aten::mat4::Identity);
        scene->add(box);
    }
}

void AlphaBlendedObjCornellBoxScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
    pos = aten::vec3(0.f, 1.f, 3.f);
    at = aten::vec3(0.f, 1.f, 0.f);
    fov = 45.0f;
}

/////////////////////////////////////////////////////

void CryteckSponzaScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    auto objs = aten::ObjLoader::load(
        "../../asset/crytek_sponza/sponza.obj", ctxt,
        [&](std::string_view name, aten::context& ctxt,
            aten::MaterialType type, const aten::vec3& mtrl_clr,
            const std::string& albedo, const std::string& nml) -> auto {
                auto albedo_map = albedo.empty()
                    ? nullptr
                    : aten::ImageLoader::load("../../asset/crytek_sponza/" + albedo, ctxt);
                auto nml_map = nml.empty()
                    ? nullptr
                    : aten::ImageLoader::load("../../asset/crytek_sponza/" + nml, ctxt);

                auto mtrl = CreateMaterial(name, ctxt, type, mtrl_clr, albedo_map.get(), nml_map.get());
                return mtrl;
        });

    objs[0]->importInternalAccelTree("../../asset/crytek_sponza/sponza.sbvh", ctxt, 0);

    auto sponza = aten::TransformableFactory::createInstance<aten::PolygonObject>(ctxt, objs[0], aten::mat4::Identity);

    scene->add(sponza);
}

void CryteckSponzaScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
#if 0
    pos = aten::vec3(0.f, 1.f, 3.f);
    at = aten::vec3(0.f, 1.f, 0.f);
#else
    pos = aten::vec3(-354.4f, 359.6f, -41.2f);
    at = aten::vec3(-353.4f, 359.4f, -41.2f);
#endif
    fov = 45.0f;
}

/////////////////////////////////////////////////////

void ManyLightCryteckSponzaScene::makeScene(aten::context& ctxt, aten::scene* scene)
{
    CryteckSponzaScene::makeScene(ctxt, scene);

    constexpr int32_t step = 5;

    const auto& aabb = scene->getBoundingBox();
    const auto& min_pos = aabb.minPos();
    const auto& max_pos = aabb.maxPos();

    // NOTE:
    // The unit seems to be [cm] not [m]...
    aten::vec3 step_v = max_pos - min_pos;
    step_v /= step;

    aten::vec3 pos = min_pos;

    std::array<aten::vec3, 3> color = {
        aten::vec3(1.0f, 0.0f, 0.0f),
        aten::vec3(0.0f, 1.0f, 0.0f),
        aten::vec3(0.0f, 0.0f, 1.0f),
    };

    size_t num = 0;

    for (int32_t y = 0; y < step; y++) {
        pos.z = 0.0f;
        for (int32_t z = 0; z < step; z++) {
            pos.x = 0.0f;
            for (int32_t x = 0; x < step; x++) {
                auto l = std::make_shared<aten::PointLight>(
                    pos,
                    color[num % color.size()],
                    6000.0f);

                // NOTE:
                // Scaling for unit is [cm].
                l->param().scale = 100.0f;

                ctxt.AddLight(l);

                pos.x += step_v.x;
                num++;
            }
            pos.z += step_v.z;
        }
        pos.y += step_v.y;
    }

    auto l = std::make_shared<aten::PointLight>(
        aten::vec3(-353.4f, 359.4f, -41.2f),
        aten::vec3(1.0f, 1.0f, 1.0f),
        6000.0f);

    // NOTE:
    // Scaling for unit is [cm].
    l->param().scale = 100.0f;

    ctxt.AddLight(l);
}

void ManyLightCryteckSponzaScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
    pos = aten::vec3(-354.4f, 359.6f, -41.2f);
    at = aten::vec3(-353.4f, 359.4f, -41.2f);
    fov = 45.0f;
}

/////////////////////////////////////////////////////

void CornellBoxSmokeScene::makeScene(
    aten::context& ctxt, aten::scene* scene)
{
    auto emit = CreateMaterial("light", ctxt, aten::MaterialType::Emissive, aten::vec3(1.0f, 1.0f, 1.0f));

    auto objs = aten::ObjLoader::load("../../asset/cornellbox/box_smoke.obj", ctxt,
        [&](std::string_view name, aten::context& ctxt,
            aten::MaterialType type, const aten::vec3& mtrl_clr,
            const std::string& albedo, const std::string& nml) -> auto {
        (void)albedo;
        (void)nml;

        if (name == "medium") {
            auto mtrl_param = AT_NAME::HomogeniousMedium::CreateMaterialParameter(
                -0.4F,
                0.0F, 0.5F,
                aten::vec3(1.0F, 0.0F, 0.0F));

            auto mtrl = ctxt.CreateMaterialWithMaterialParameter(
                name,
                mtrl_param,
                nullptr, nullptr, nullptr);
            return mtrl;
        }
        else {
            auto mtrl = CreateMaterial(name, ctxt, type, mtrl_clr);
            return mtrl;
        }
    },
        true, true);

    auto light = aten::TransformableFactory::createInstance<aten::PolygonObject>(
        ctxt,
        objs[0],
        aten::vec3(0.0f),
        aten::vec3(0.0f),
        aten::vec3(1.0f));
    scene->add(light);

    auto areaLight = std::make_shared<aten::AreaLight>(light, emit->param().baseColor, 20.0f);
    ctxt.AddLight(areaLight);

    for (int32_t i = 1; i < objs.size(); i++) {
        auto obj = aten::TransformableFactory::createInstance<aten::PolygonObject>(ctxt, objs[i], aten::mat4::Identity);
        scene->add(obj);
    }
}

void CornellBoxSmokeScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
    pos = aten::vec3(0.f, 1.f, 3.f);
    at = aten::vec3(0.f, 1.f, 0.f);
    fov = 45.0f;
}

/////////////////////////////////////////////////////

void CornellBoxHomogeneousMediumScene::makeScene(
    aten::context& ctxt, aten::scene* scene)
{
    auto emit = CreateMaterial("light", ctxt, aten::MaterialType::Emissive, aten::vec3(1.0f, 1.0f, 1.0f));

    auto objs = aten::ObjLoader::load("../../asset/cornellbox/orig.obj", ctxt,
        [&](std::string_view name, aten::context& ctxt,
            aten::MaterialType type, const aten::vec3& mtrl_clr,
            const std::string& albedo, const std::string& nml) -> auto {
        (void)albedo;
        (void)nml;

        if (name == "shortBox") {
            auto mtrl_param = AT_NAME::HomogeniousMedium::CreateMaterialParameter(
                -0.4F,
                0.5F, 0.5F,
                aten::vec3(1.0F, 0.0F, 0.0F));

            auto mtrl = ctxt.CreateMaterialWithMaterialParameter(
                name,
                mtrl_param,
                nullptr, nullptr, nullptr);
            return mtrl;
        }
        else if (name == "tallBox") {
            auto mtrl_param = AT_NAME::HomogeniousMedium::CreateMaterialParameter(
                -0.4F,
                0.5F, 0.5F,
                aten::vec3(0.0F, 1.0F, 0.0F));

            auto mtrl = ctxt.CreateMaterialWithMaterialParameter(
                name,
                mtrl_param,
                nullptr, nullptr, nullptr);
            return mtrl;
        }
        else {
            auto mtrl = CreateMaterial(name, ctxt, type, mtrl_clr);
            return mtrl;
        }
    },
        true, true);

    auto light = aten::TransformableFactory::createInstance<aten::PolygonObject>(
        ctxt,
        objs[0],
        aten::vec3(0.0f),
        aten::vec3(0.0f),
        aten::vec3(1.0f));
    scene->add(light);

    auto areaLight = std::make_shared<aten::AreaLight>(light, emit->param().baseColor, 20.0f);
    ctxt.AddLight(areaLight);

    for (int32_t i = 1; i < objs.size(); i++) {
        auto obj = aten::TransformableFactory::createInstance<aten::PolygonObject>(ctxt, objs[i], aten::mat4::Identity);
        scene->add(obj);
    }
}

void CornellBoxHomogeneousMediumScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
    pos = aten::vec3(0.f, 1.f, 3.f);
    at = aten::vec3(0.f, 1.f, 0.f);
    fov = 45.0f;
}

/////////////////////////////////////////////////////

void HomogeneousMediumRefractionBunnyScene::makeScene(
    aten::context& ctxt, aten::scene* scene)
{
    auto emit = CreateMaterial("light", ctxt, aten::MaterialType::Emissive, aten::vec3(1.0f, 1.0f, 1.0f));

    auto objs = aten::ObjLoader::load("../../asset/cornellbox/bunny_in_box.obj", ctxt,
        [&](std::string_view name, aten::context& ctxt,
            aten::MaterialType type, const aten::vec3& mtrl_clr,
            const std::string& albedo, const std::string& nml) -> auto {
        (void)albedo;
        (void)nml;

        if (name == "material_0") {
            aten::MaterialParameter mtrl_param;
            mtrl_param.type = aten::MaterialType::Refraction;
#ifdef WHITE_FURNACE_TEST
            mtrl_param.baseColor = aten::vec3(1.0F);
#else
            mtrl_param.baseColor = aten::vec3(0.580000f, 0.580000f, 0.580000f);
#endif
            mtrl_param.standard.ior = 1.333F;

            auto mtrl = ctxt.CreateMaterialWithMaterialParameter(
                name,
                mtrl_param,
                nullptr, nullptr, nullptr);

            mtrl->param().is_medium = true;
            mtrl->param().medium.phase_function_g = 0.4F;
            mtrl->param().medium.sigma_a = 0.0F;
            mtrl->param().medium.sigma_s = 0.9F;
            mtrl->param().medium.le = aten::vec3(0.8F);
            return mtrl;
        }
        else {
            auto mtrl = CreateMaterial(name, ctxt, type, mtrl_clr);
            return mtrl;
        }
    },
        true, true);

    auto light = aten::TransformableFactory::createInstance<aten::PolygonObject>(
        ctxt,
        objs[0],
        aten::vec3(0.0f),
        aten::vec3(0.0f),
        aten::vec3(1.0f));
    scene->add(light);

    auto areaLight = std::make_shared<aten::AreaLight>(light, emit->param().baseColor, 20.0f);
    ctxt.AddLight(areaLight);

    for (int32_t i = 1; i < objs.size(); i++) {
        auto obj = aten::TransformableFactory::createInstance<aten::PolygonObject>(ctxt, objs[i], aten::mat4::Identity);
        scene->add(obj);
    }
}

void HomogeneousMediumRefractionBunnyScene::getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    float& fov)
{
    pos = aten::vec3(0.f, 1.f, 3.f);
    at = aten::vec3(0.f, 1.f, 0.f);
    fov = 45.0f;
}
