#include "tinyxml2.h"

#include "SceneLoader.h"
#include "utility.h"
#include "ImageLoader.h"
#include "MaterialLoader.h"
#include "ObjLoader.h"
#include "AssetManager.h"

namespace aten
{
    static std::string g_base;

    void SceneLoader::setBasePath(const std::string& base)
    {
        g_base = removeTailPathSeparator(base);
    }

    // NOTE
    // <scene width=<uint> height=<uint>>
    //        <camera 
    //            type=<string>
    //            org=<vec3>
    //            at=<vec3> 
    //            vfov=<real>
    //            sensorsize=<real>
    //            dist_sensor_lens=<real>
    //            dist_lens_focus=<real> 
    //            lens_r=<real>
    //            w_scale=<real>
    //        />
    //        <renderer type=<string> spp=<uint> mutaion=<uint> mlt=<uint> depth=<uint> rrdepth=<uint>/>
    //        <materials>
    //            <material path=<string/>
    //            <material [attributes...]/>
    //        </materials>
    //        <textures>
    //            <texture name=<string> path=<string>/>
    //        </textures>
    //        <objects>
    //            <object name=<string> type="object" path=<string> trans=<vec3> rotate=<vec3> scale=<real> material=<string>/>
    //            <object name=<string> type="sphere" center=<vec3> radius=<real> material=<string>/>
    //            <object name=<string> type="cube" center=<vec3> width=<real> height=<real> depth=<real> material=<string>/>
    //        </objects>
    //        <lights>
    //            <light type=<string> color=<vec3> [attributes...]/>
    //        </lights>
    //        <preprocs>
    //            <proc type=<string> [attributes...]/>
    //        </preprocs>
    //        <postprocs>
    //            <proc type=<string> [attributes...]/>
    //        </postprocs>
    // </scene>

    void readTextures(const tinyxml2::XMLElement* root, aten::context& ctxt)
    {
        auto texRoot = root->FirstChildElement("textures");

        if (!texRoot) {
            return;
        }

        for (auto elem = texRoot->FirstChildElement("texture"); elem != nullptr; elem = elem->NextSiblingElement("texture")) {
            std::string path;
            std::string tag;

            for (auto attr = elem->FirstAttribute(); attr != nullptr; attr = attr->Next()) {
                std::string attrName(attr->Name());

                if (attrName == "path") {
                    path = attr->Value();
                }
                else if (attrName == "name") {
                    tag = attr->Value();
                }
            }

            if (!tag.empty()) {
                ImageLoader::load(tag, path, ctxt);
            }
            else {
                ImageLoader::load(path, ctxt);
            }
        }
    }

    void SceneLoader::readMaterials(
        const void* xmlRoot,
        context& ctxt)
    {
        const tinyxml2::XMLElement* root = reinterpret_cast<const tinyxml2::XMLElement*>(xmlRoot);

        auto mtrlRoot = root->FirstChildElement("materials");
        
        if (!mtrlRoot) {
            return;
        }

        for (auto elem = mtrlRoot->FirstChildElement("material"); elem != nullptr; elem = elem->NextSiblingElement("material")) {
            std::string path;
            std::string tag;

            tinyxml2::XMLDocument xml;
            tinyxml2::XMLElement* root = nullptr;
            tinyxml2::XMLElement* mtrlElem = nullptr;

            for (auto attr = elem->FirstAttribute(); attr != nullptr; attr = attr->Next()) {
                std::string attrName(attr->Name());

                if (attrName == "path") {
                    path = attr->Value();
                }
                else {
                    if (attrName == "name") {
                        tag = attr->Value();
                    }

                    if (!root) {
                        root = xml.NewElement("root");
                        xml.InsertEndChild(root);
                    }

                    if (!mtrlElem) {
                        mtrlElem = xml.NewElement("material");
                        root->InsertEndChild(mtrlElem);
                    }

                    mtrlElem->SetAttribute(attr->Name(), attr->Value());
                }
            }

            if (mtrlElem) {
                root->InsertEndChild(mtrlElem);
                MaterialLoader::onLoad(root, ctxt);
            }
            else if (!path.empty()) {
                MaterialLoader::load(path, ctxt);
            }
        }
    }


    template <typename TYPE>
    aten::PolymorphicValue getValue(const tinyxml2::XMLAttribute* a)
    {
        AT_ASSERT(false);
        PolymorphicValue ret;
        return ret;
    }

    template <>
    aten::PolymorphicValue getValue<vec3>(const tinyxml2::XMLAttribute* a)
    {
        aten::PolymorphicValue v;

        std::string text(a->Value());

        std::vector<std::string> values;
        int num = split(text, values, ' ');

        for (int i = 0; i < std::min<int>(num, 3); i++) {
            v.val.v[i] = (real)atof(values[i].c_str());
        }

        return std::move(v);
    }

    template <>
    aten::PolymorphicValue getValue<real>(const tinyxml2::XMLAttribute* a)
    {
        aten::PolymorphicValue v;
        v.val.f = (real)a->DoubleValue();
        return std::move(v);
    }

    template <>
    aten::PolymorphicValue getValue<int>(const tinyxml2::XMLAttribute* a)
    {
        aten::PolymorphicValue v;
        v.val.i = a->IntValue();
        return std::move(v);
    }

    material* findMaterialInObject(
        const tinyxml2::XMLElement* root,
        const std::string& objtag)
    {
        auto objRoot = root->FirstChildElement("objects");

        if (!objRoot) {
            return nullptr;
        }

        material* mtrl = nullptr;

        for (auto elem = objRoot->FirstChildElement("object"); elem != nullptr; elem = elem->NextSiblingElement("object")) {
            std::string tag;

            for (auto attr = elem->FirstAttribute(); attr != nullptr; attr = attr->Next()) {
                std::string attrName(attr->Name());

                if (attrName == "name") {
                    tag = attr->Value();        
                }
                else if (attrName == "material") {
                    mtrl = AssetManager::getMtrl(attr->Value());
                }
            }

            if (objtag == tag) {
                break;
            }
            else {
                mtrl = nullptr;
            }
        }

        return mtrl;
    }

    void readObjects(
        const tinyxml2::XMLElement* root,
        context& ctxt,
        std::map<std::string, transformable*>& objs)
    {
        auto objRoot = root->FirstChildElement("objects");

        if (!objRoot) {
            return;
        }

        for (auto elem = objRoot->FirstChildElement("object"); elem != nullptr; elem = elem->NextSiblingElement("object")) {
            std::string path;
            std::string tag;
            std::string type;

            Values val;

            material* mtrl = nullptr;

            for (auto attr = elem->FirstAttribute(); attr != nullptr; attr = attr->Next()) {
                std::string attrName(attr->Name());

                if (attrName == "path") {
                    path = attr->Value();
                }
                else if (attrName == "name") {
                    tag = attr->Value();
                }
                else if (attrName == "type") {
                    type = attr->Value();
                }
                else if (attrName == "trans" || attrName == "rotate" || attrName == "center") {
                    auto v = getValue<vec3>(attr);
                    val.add(attrName, v);
                }
                else if (attrName == "scale"
                    || attrName == "radius"
                    || attrName == "width" 
                    || attrName == "height" 
                    || attrName == "depth")
                {
                    auto v = getValue<real>(attr);
                    val.add(attrName, v);
                }
                else if (attrName == "material") {
                    mtrl = AssetManager::getMtrl(attr->Value());
                }
            }

            if (tag.empty()) {
                // TODO
                // throw exception.
                throw new std::exception();
            }

            object* obj = nullptr;

            if (type == "object") {
                obj = ObjLoader::load(path, ctxt);
            }

            mat4 mtxS;
            mtxS.asScale(val.get("scale", real(1)));

            mat4 mtxRotX, mtxRotY, mtxRotZ;
            auto rotate = val.get("rotate", vec3(0));
            mtxRotX.asRotateByX(rotate.x);
            mtxRotY.asRotateByY(rotate.x);
            mtxRotZ.asRotateByZ(rotate.x);

            mat4 mtxT;
            mtxT.asTrans(val.get("trans", vec3(0)));

            auto mtxL2W = mtxT * mtxRotX * mtxRotY * mtxRotZ * mtxS;

            if (obj) {
                auto instance = aten::TransformableFactory::createInstance<aten::object>(ctxt, obj, mtxL2W);
                objs.insert(std::pair<std::string, transformable*>(tag, instance));
            }
            else {
                if (type == "cube") {
                    auto cube = aten::TransformableFactory::createCube(
                        ctxt,
                        val.get("center", vec3(0)),
                        val.get("width", real(1)),
                        val.get("height", real(1)),
                        val.get("depth", real(1)),
                        mtrl);
                    objs.insert(std::pair<std::string, transformable*>(tag, cube));
                }
                else if (type == "sphere") {
                    auto sphere = aten::TransformableFactory::createSphere(
                        ctxt,
                        val.get("center", vec3(0)),
                        val.get("radius", real(1)),
                        mtrl);
                    objs.insert(std::pair<std::string, transformable*>(tag, sphere));
                }
                else {
                    // TODO
                    // warning.
                }
            }
        }
    }

    void readLights(
        const tinyxml2::XMLElement* root,
        const std::map<std::string, transformable*>& objs,
        std::vector<Light*>& lights)
    {
        auto lightRoot = root->FirstChildElement("lights");

        if (!lightRoot) {
            return;
        }

        for (auto elem = lightRoot->FirstChildElement("light"); elem != nullptr; elem = elem->NextSiblingElement("light")) {
            std::string type;
            Values val;
            std::string objtag;

            for (auto attr = elem->FirstAttribute(); attr != nullptr; attr = attr->Next()) {
                std::string attrName(attr->Name());

                if (attrName == "type") {
                    type = attr->Value();
                }
                else if (attrName == "color" || attrName == "pos" || attrName == "dir") {
                    auto v = getValue<vec3>(attr);
                    val.add(attrName, v);
                }
                else if (attrName == "envmap") {
                    auto envmap = AssetManager::getTex(attr->Value());
                    if (envmap) {
                        PolymorphicValue v;
                        v.val.p = envmap;
                        val.add(attrName, v);
                    }
                }
                else if (attrName == "object") {
                    objtag = attr->Value();
                }
                else {
                    auto v = getValue<real>(attr);
                    val.add(attrName, v);
                }
            }

            Light* light = nullptr;

            if (type == "area") {
                if (!objtag.empty()) {
                    auto it = objs.find(objtag);
                    if (it != objs.end()) {
                        auto mtrl = findMaterialInObject(root, objtag);
                        if (mtrl) {
                            light = new AreaLight(it->second, mtrl->color());
                        }
                    }
                }
            }
            else if (type == "point") {
                light = new PointLight(val);
            }
            else if (type == "spot") {
                light = new SpotLight(val);
            }
            else if (type == "dir") {
                light = new DirectionalLight(val);
            }
            else if (type == "ibl") {
                light = new ImageBasedLight(val);
            }

            if (light) {
                lights.push_back(light);
            }
        }
    }

    void readProcs(
        const tinyxml2::XMLElement* root,
        const std::string& elemName,
        std::vector<SceneLoader::ProcInfo>& infos)
    {
        auto procRoot = root->FirstChildElement(elemName.c_str());

        if (!procRoot) {
            return;
        }

        for (auto elem = procRoot->FirstChildElement("proc"); elem != nullptr; elem = elem->NextSiblingElement("proc")) {
            infos.push_back(SceneLoader::ProcInfo());

            auto& info = infos[infos.size() - 1];

            for (auto attr = elem->FirstAttribute(); attr != nullptr; attr = attr->Next()) {
                std::string attrName(attr->Name());

                if (attrName == "type") {
                    info.type = attr->Value();
                }
                else {
                    auto v = getValue<real>(attr);
                    info.val.add(attrName, v);
                }
            }
        }
    }

    camera* readCamera(
        const tinyxml2::XMLElement* root,
        uint32_t width, uint32_t height)
    {
        auto camRoot = root->FirstChildElement("camera");

        if (!camRoot) {
            // TODO
            // throw exception.
            throw new std::exception();
        }

        std::string type;
        Values val;

        for (auto attr = camRoot->FirstAttribute(); attr != nullptr; attr = attr->Next()) {
            std::string attrName(attr->Name());

            if (attrName == "type") {
                type = attr->Value();
            }
            else if (attrName == "org" || attrName == "at" || attrName == "up") {
                auto v = getValue<vec3>(attr);
                val.add(attrName, v);
            }
            else {
                auto v = getValue<real>(attr);
                val.add(attrName, v);
            }
        }

        camera* cam = nullptr;

        if (type == "pinhole") {
            auto pinhole = new PinholeCamera();
            pinhole->init(
                val.get("org", vec3(0)),
                val.get("at", vec3(0, 0, -1)),
                val.get("up", vec3(0, 1, 0)),
                val.get("vfov", real(30)),
                width, height);
            cam = pinhole;
        }
        else if (type == "thinlens") {
            auto thinlens = new ThinLensCamera();
            thinlens->init(
                width, height,
                val.get("org", vec3(0)),
                val.get("at", vec3(0, 0, -1)),
                val.get("up", vec3(0, 1, 0)),
                val.get("sensorsize", real(30.0)),
                val.get("dist_sensor_lens", real(40.0)),
                val.get("dist_lens_focus", real(130.0)),
                val.get("lens_r", real(1.0)),
                val.get("w_scale", real(1.0)));
            cam = thinlens;
        }
        else if (type == "equirect") {
            auto equirect = new EquirectCamera();
            equirect->init(
                val.get("org", vec3(0)),
                val.get("at", vec3(0, 0, -1)),
                val.get("up", vec3(0, 1, 0)),
                width, height);
            cam = equirect;
        }

        return cam;
    }

    void readRenderParams(
        const tinyxml2::XMLElement* root,
        SceneLoader::SceneInfo& info)
    {
        auto renderRoot = root->FirstChildElement("renderer");

        if (!renderRoot) {
            // TODO
            // throw exception.
            throw new std::exception();
        }

        Values val;

        for (auto attr = renderRoot->FirstAttribute(); attr != nullptr; attr = attr->Next()) {
            std::string attrName(attr->Name());

            if (attrName == "type") {
                info.rendererType = attr->Value();
            }
            else {
                auto v = getValue<int>(attr);
                val.add(attrName, v);
            }
        }

        info.dst.sample = val.get("spp", int(1));
        info.dst.maxDepth = val.get("depth", int(5));
        info.dst.russianRouletteDepth = val.get("rrdepth", int(3));
        info.dst.mutation = val.get("mutation", int(100));
        info.dst.mltNum = val.get("mlt", int(100));
    }

    SceneLoader::SceneInfo SceneLoader::load(
        const std::string& path,
        context& ctxt)
    {
        std::string fullpath = path;
        if (!g_base.empty()) {
            fullpath = g_base + "/" + fullpath;
        }

        tinyxml2::XMLDocument xml;
        auto err = xml.LoadFile(fullpath.c_str());
        if (err != tinyxml2::XML_SUCCESS) {
            // TODO
            // throw exception.
            throw new std::exception();
        }

        SceneLoader::SceneInfo ret;
        {
            ret.dst.width = 0;
            ret.dst.height = 0;
        }

        std::map<std::string, transformable*> objs;
        std::vector<Light*> lights;

        auto root = xml.FirstChildElement("scene");
        if (!root) {
            // TODO
            // throw exception.
            throw new std::exception();
        }
        else {
            for (auto attr = root->FirstAttribute(); attr != nullptr; attr = attr->Next()) {
                std::string attrName(attr->Name());

                if (attrName == "width") {
                    auto v = getValue<int>(attr);
                    ret.dst.width = v.getAs<int>();
                }
                else if (attrName == "height") {
                    auto v = getValue<int>(attr);
                    ret.dst.height = v.getAs<int>();
                }
            }

            if (ret.dst.width == 0 || ret.dst.height == 0) {
                // TODO
                // throw exception.
                throw new std::exception();
            }

            readRenderParams(root, ret);
            
            ret.camera = readCamera(root, ret.dst.width, ret.dst.height);

            readTextures(root, ctxt);
            readMaterials(root, ctxt);
            readObjects(root, ctxt, objs);
            readLights(root, objs, lights);
            readProcs(root, "preprocs", ret.preprocs);
            readProcs(root, "postprocs", ret.postprocs);
        }

        // TODO
        ret.scene = new aten::AcceleratedScene<aten::bvh>();

        for (auto it = objs.begin(); it != objs.end(); it++) {
            auto obj = it->second;

            ret.scene->add(obj);
        }

        for (auto it = lights.begin(); it != lights.end(); it++) {
            auto light = *it;
            if (light->isIBL()) {
                // TODO
                ret.scene->addImageBasedLight((ImageBasedLight*)light);
            }
            else {
                ret.scene->addLight(light);
            }
        }

        return std::move(ret);
    }
}
