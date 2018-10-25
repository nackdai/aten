#include "renderer/bdpt.h"
#include "misc/omputil.h"
#include "misc/timer.h"
#include "material/lambert.h"
#include "material/refraction.h"
#include "geometry/transformable.h"
#include "sampler/xorshift.h"
#include "sampler/halton.h"
#include "sampler/sobolproxy.h"
#include "sampler/cmj.h"

//#define BDPT_DEBUG

#ifdef BDPT_DEBUG
#pragma optimize( "", off)
#endif

namespace aten
{
    static inline real russianRoulette(const vec3& v)
    {
        real p = std::max(v.r, std::max(v.g, v.b));
        p = aten::clamp(p, real(0), real(1));
        return p;
    }

    static inline real russianRoulette(const material* mtrl)
    {
        if (mtrl->isEmissive()) {
            return 1;
        }
        real p = russianRoulette(mtrl->color());
        return p;
    }

    real BDPT::russianRoulette(const Vertex& vtx)
    {
        real pdf = real(0);

        if (vtx.mtrl) {
            pdf = aten::russianRoulette(vtx.mtrl);
        }
        else if (vtx.light) {
            pdf = aten::russianRoulette(vtx.light->getLe());
        }
        else {
            pdf = real(0);
        }

        return pdf;
    }

    BDPT::Result BDPT::genEyePath(
        const context& ctxt,
        std::vector<Vertex>& vs,
        int x, int y,
        sampler* sampler,
        scene* scene,
        camera* camera) const
    {
        real u = (real)(x + sampler->nextSample()) / (real)m_width;
        real v = (real)(y + sampler->nextSample()) / (real)m_height;

        auto camsample = camera->sample(u, v, sampler);

        // レンズ上の頂点（x0）を頂点リストに追加.
        vs.push_back(Vertex(
            camsample.posOnLens,
            camsample.nmlOnLens,
            camsample.nmlOnLens,
            ObjectType::Lens,
            camsample.pdfOnLens,
            vec3(1),
            vec3(0),
            nullptr,
            nullptr,
            0, 0));

        int depth = 0;

        ray ray = camsample.r;

        vec3 throughput = vec3(1);
        real totalAreaPdf = camsample.pdfOnLens;

        vec3 prevNormal = camera->getDir();
        real sampledPdf = real(1);

        //while (depth < m_maxDepth) {
        for (;;) {
            hitrecord rec;
            Intersection isect;
            if (!scene->hit(ctxt, ray, AT_MATH_EPSILON, AT_MATH_INF, rec, isect)) {
                break;
            }

            // 交差位置の法線.
            // 物体からのレイの入出を考慮.
            vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

            auto mtrl = ctxt.getMaterial(rec.mtrlid);
            auto obj = ctxt.getTransformable(isect.objid);

            // ロシアンルーレットによって、新しい頂点を「実際に」サンプリングし、生成するのかどうかを決定する.
            auto rrProb = aten::russianRoulette(mtrl);
            auto rr = sampler->nextSample();
            if (rr >= rrProb) {
                break;
            }

            // 新しい頂点がサンプリングされたので、トータルの確率密度に乗算する.
            totalAreaPdf *= rrProb;

            const vec3 toNextVtx = ray.org - rec.p;

            if (depth == 0) {
                // NOTE
                // レンダリング方程式２.
                // http://rayspace.xyz/CG/contents/LTE2.html

                // x1のサンプリング確率密度はイメージセンサ上のサンプリング確率密度を変換することで求める,
                auto pdfOnImageSensor = camera->convertImageSensorPdfToScenePdf(
                    camsample.pdfOnImageSensor,
                    rec.p,
                    orienting_normal,
                    camsample.posOnImageSensor,
                    camsample.posOnLens,
                    camsample.posOnObjectplane);

                totalAreaPdf *= pdfOnImageSensor;

                // 幾何的な係数計算 + センサーセンシティビティの項を計算する
                // x1 -> x0への放射輝度が最終的にイメージセンサに与える寄与度
                auto W_dash = camera->getWdash(
                    rec.p,
                    orienting_normal,
                    camsample.posOnImageSensor,
                    camsample.posOnLens,
                    camsample.posOnObjectplane);

                throughput = W_dash * throughput;
            }
            else {
                // 新しい頂点をサンプリングするための確率密度関数は立体角測度に関するものであったため、これを面積速度に関する確率密度関数に変換する.
                const real c = dot(normalize(toNextVtx), orienting_normal);
                const real dist2 = squared_length(toNextVtx);
                const real areaPdf = sampledPdf * (c / dist2);

                totalAreaPdf *= areaPdf;
            }

            if (depth == 0 && camera->isPinhole()) {
                // Nothing is done...
            }
            else {
                // ジオメトリターム.
                const real c0 = dot(normalize(toNextVtx), orienting_normal);
                const real c1 = dot(normalize(-toNextVtx), prevNormal);
                const real dist2 = squared_length(toNextVtx);
                const real G = c0 * c1 / dist2;
                throughput = G * throughput;
            }

            // 光源にヒットしたらそこで追跡終了.
            if (mtrl->isEmissive()) {
                vec3 bsdf = lambert::bsdf(&mtrl->param(), rec.u, rec.v);

                vs.push_back(Vertex(
                    rec.p,
                    rec.normal,
                    orienting_normal,
                    ObjectType::Light,
                    totalAreaPdf,
                    throughput,
                    bsdf,
                    obj,
                    mtrl,
                    rec.u, rec.v));

                vec3 emit = mtrl->color();
                vec3 contrib = throughput * emit / totalAreaPdf;

                return std::move(Result(contrib, x, y, true));
            }

            auto sampling = mtrl->sample(ray, orienting_normal, rec.normal, sampler, rec.u, rec.v);

            sampledPdf = sampling.pdf;
            auto sampledBsdf = sampling.bsdf;

            if (mtrl->isSingular()) {
                // For canceling probabaility to select reflection or rafraction.
                sampledBsdf *= sampling.subpdf;

                // For canceling cosine term.
                auto costerm = dot(normalize(toNextVtx), orienting_normal);
                sampledBsdf /= costerm;
            }

            // 新しい頂点を頂点リストに追加する.
            vs.push_back(Vertex(
                rec.p,
                rec.normal,
                orienting_normal,
                ObjectType::Object,
                totalAreaPdf,
                throughput,
                sampling.bsdf,
                obj,
                mtrl,
                rec.u, rec.v));

            throughput *= sampledBsdf;
            
            // refractionの反射、屈折の確率を掛け合わせる.
            // refraction以外では 1 なので影響はない.
            totalAreaPdf *= sampling.subpdf;

            vec3 nextDir = normalize(sampling.dir);

            ray = aten::ray(rec.p + nextDir * AT_MATH_EPSILON, nextDir);

            prevNormal = orienting_normal;
            depth++;
        }

        return std::move(Result(vec3(), -1, -1, false));
    }

    BDPT::Result BDPT::genLightPath(
        const context& ctxt,
        std::vector<Vertex>& vs,
        aten::Light* light,
        sampler* sampler,
        scene* scene,
        camera* camera) const
    {
        // TODO
        // Only AreaLight...

        // 光源上にサンプル点生成（y0）.
        aten::hitable::SamplePosNormalPdfResult res;
        light->getSamplePosNormalArea(ctxt, &res, sampler);
        auto posOnLight = res.pos;
        auto nmlOnLight = res.nml;
        auto pdfOnLight = real(1) / res.area;

        // 確率密度の積を保持（面積測度に関する確率密度）.
        auto totalAreaPdf = pdfOnLight;

        // 光源上に生成された頂点を頂点リストに追加.
        vs.push_back(Vertex(
            posOnLight,
            nmlOnLight,
            nmlOnLight,
            ObjectType::Light,
            totalAreaPdf,
            vec3(real(0)),
            light->getLe(),
            light));

        // 現在の放射輝度（モンテカルロ積分のスループット）.
        // 本当は次の頂点（y1）が決まらないと光源からその方向への放射輝度値は決まらないが、今回は完全拡散光源を仮定しているので、方向に依らずに一定の値になる.
        vec3 throughput = light->getLe();

        int depth = 0;

        // 完全拡散光源を仮定しているので、Diffuse面におけるサンプリング方法と同じものをつかって次の方向を決める.
        nmlOnLight = normalize(nmlOnLight);
        vec3 dir = lambert::sampleDirection(nmlOnLight, sampler);
        real sampledPdf = lambert::pdf(nmlOnLight, dir);
        vec3 prevNormal = nmlOnLight;

        ray ray = aten::ray(posOnLight + dir * AT_MATH_EPSILON, dir);

        //while (depth < m_maxDepth) {
        for (;;) {
            hitrecord rec;
            Intersection isect;
            bool isHit = scene->hit(ctxt, ray, AT_MATH_EPSILON, AT_MATH_INF, rec, isect);

            if (!camera->isPinhole()) {
                // The light will never hit to the pinhole camera.
                // The pihnole camera lens is tooooooo small.

                vec3 posOnImageSensor;
                vec3 posOnLens;
                vec3 posOnObjectPlane;
                int pixelx;
                int pixely;

                // レンズと交差判定.
                auto lens_t = camera->hitOnLens(
                    ray,
                    posOnLens,
                    posOnObjectPlane,
                    posOnImageSensor,
                    pixelx, pixely);

                if (AT_MATH_EPSILON < lens_t && lens_t < isect.t) {
                    // レイがレンズにヒット＆イメージセンサにヒット.

                    pixelx = aten::clamp(pixelx, 0, m_width - 1);
                    pixely = aten::clamp(pixely, 0, m_height - 1);

                    vec3 dir = ray.org - posOnLens;
                    const real dist2 = squared_length(dir);
                    dir = normalize(dir);

                    const vec3& camnml = camera->getDir();

                    // レンズの上の点のサンプリング確率を計算。
                    {
                        const real c = dot(dir, camnml);
                        const real areaPdf = sampledPdf * c / dist2;

                        totalAreaPdf *= areaPdf;
                    }

                    // ジオメトリターム
                    {
                        const real c0 = dot(dir, camnml);
                        const real c1 = dot(-dir, prevNormal);
                        const real G = c0 * c1 / dist2;

                        throughput *= G;
                    }

                    // レンズ上に生成された点を頂点リストに追加（基本的に使わない）.
                    vs.push_back(Vertex(
                        posOnLens,
                        camnml,
                        camnml,
                        ObjectType::Lens,
                        totalAreaPdf,
                        throughput,
                        vec3(real(0)),
                        nullptr,
                        nullptr,
                        real(0), real(0)));

                    const real W_dash = camera->getWdash(
                        ray.org,
                        vec3(real(0), real(1), real(0)),    // pinholeのときはここにこない.また、thinlensのときは使わないので、適当な値でいい.
                        posOnImageSensor,
                        posOnLens,
                        posOnObjectPlane);

                    const vec3 contrib = throughput * W_dash / totalAreaPdf;

                    return std::move(Result(contrib, pixelx, pixely, true));
                }
            }

            if (!isHit) {
                break;
            }

            // 交差位置の法線.
            // 物体からのレイの入出を考慮.
            vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

            auto mtrl = ctxt.getMaterial(rec.mtrlid);
            auto obj = ctxt.getTransformable(isect.objid);

            // ロシアンルーレットによって、新しい頂点を「実際に」サンプリングし、生成するのかどうかを決定する.
            auto rrProb = aten::russianRoulette(mtrl);
            auto rr = sampler->nextSample();
            if (rr >= rrProb) {
                break;
            }

            // 新しい頂点がサンプリングされたので、トータルの確率密度に乗算する.
            totalAreaPdf *= rrProb;

            const vec3 toNextVtx = ray.org - rec.p;

            {
                // 新しい頂点をサンプリングするための確率密度関数は立体角測度に関するものであったため、これを面積測度に関する確率密度関数に変換する.
                const real c = dot(normalize(toNextVtx), orienting_normal);
                const real dist2 = squared_length(toNextVtx);
                const real areaPdf = sampledPdf * (c / dist2);

                // 全ての頂点をサンプリングする確率密度の総計を出す。
                totalAreaPdf *= areaPdf;
            }

            {
                // ジオメトリターム.
                const real c0 = dot(normalize(toNextVtx), orienting_normal);
                const real c1 = dot(normalize(-toNextVtx), prevNormal);
                const real dist2 = squared_length(toNextVtx);
                const real G = c0 * c1 / dist2;

                throughput = G * throughput;
            }

            auto sampling = mtrl->sample(ray, orienting_normal, rec.normal, sampler, rec.u, rec.v, true);

            sampledPdf = sampling.pdf;
            auto sampledBsdf = sampling.bsdf;

            if (mtrl->isSingular()) {
                // For canceling probabaility to select reflection or rafraction.
                sampledBsdf *= sampling.subpdf;

                // For canceling cosine term.
                auto costerm = dot(normalize(toNextVtx), orienting_normal);
                sampledBsdf /= costerm;
            }
            else if (mtrl->isEmissive()) {
                sampledBsdf = vec3(real(0));
            }

            // 新しい頂点を頂点リストに追加する.
            vs.push_back(Vertex(
                rec.p,
                rec.normal,
                orienting_normal,
                ObjectType::Object,
                totalAreaPdf,
                throughput,
                sampledBsdf,
                obj,
                mtrl,
                rec.u, rec.v));

            throughput *= sampledBsdf;

            // refractionの反射、屈折の確率を掛け合わせる.
            // refraction以外では 1 なので影響はない.
            totalAreaPdf *= sampling.subpdf;

            vec3 nextDir = normalize(sampling.dir);

            ray = aten::ray(rec.p + nextDir * AT_MATH_EPSILON, nextDir);

            prevNormal = orienting_normal;
            depth++;
        }

        return std::move(Result(vec3(), -1, -1, false));
    }

    // 頂点fromから頂点nextをサンプリングしたとするとき、面積測度に関するサンプリング確率密度を計算する.
    real BDPT::computAreaPdf(
        camera* camera,
        const std::vector<const Vertex*>& vs,
        const int prev_idx,            // 頂点curの前の頂点のインデックス.
        const int cur_idx,          // 頂点curのインデックス.
        const int next_idx) const   // 頂点nextのインデックス.
    {
        const Vertex& curVtx = *vs[cur_idx];
        const Vertex& nextVtx = *vs[next_idx];

        Vertex const* prevVtx = nullptr;
        if (0 <= prev_idx && prev_idx < vs.size()) {
            prevVtx = vs[prev_idx];
        }

        const vec3 to = nextVtx.pos - curVtx.pos;
        const vec3 normalizedTo = normalize(to);

        real pdf = real(0);

        // 頂点fromのオブジェクトの種類によって、その次の頂点のサンプリング確率密度の計算方法が変わる.
        if (curVtx.objType == ObjectType::Light) {
            // TODO
            // Lightは完全拡散面として扱う.
            pdf = lambert::pdf(curVtx.orienting_normal, normalizedTo);
        }
        else if (curVtx.objType == ObjectType::Lens) {
            // レンズ上の点からシーン上の点をサンプリングするときの面積測度に関する確率密度を計算.
            // イメージセンサ上の点のサンプリング確率密度を元に変換する.

            if (camera->isPinhole()) {
                // TODO
                // I don't understand why.
                // But, it seems that the rendering result is correct...
                return real(1);
            }

            // シーン上の点からレンズに入るレイ.
            ray r(nextVtx.pos, -normalizedTo);

            vec3 posOnLens;
            vec3 posOnObjectplane;
            vec3 posOnImagesensor;
            int x, y;

            auto lens_t = camera->hitOnLens(
                r,
                posOnLens,
                posOnObjectplane,
                posOnImagesensor,
                x, y);

            if (lens_t > AT_MATH_EPSILON) {
                // レイがレンズにヒット＆イメージセンサにヒット.

                real imagesensorWidth = camera->getImageSensorWidth();
                real imagesensorHeight = camera->getImageSensorHeight();
                real pdfImage = real(1) / (imagesensorWidth * imagesensorHeight);

                // イメージセンサ上のサンプリング確率密度を計算.
                // イメージセンサの面積測度に関する確率密度をシーン上のサンプリング確率密度（面積測度に関する確率密度）に変換されている.
                const real imageSensorAreaPdf = camera->convertImageSensorPdfToScenePdf(
                    pdfImage,
                    nextVtx.pos,
                    nextVtx.orienting_normal,
                    posOnImagesensor,
                    posOnLens,
                    posOnObjectplane);

                return imageSensorAreaPdf;
            }
            else {
                return real(0);
            }
        }
        else {
            if (prevVtx) {
                const vec3 wi = normalize(curVtx.pos - prevVtx->pos);
                const vec3 wo = normalize(nextVtx.pos - curVtx.pos);

                if (curVtx.mtrl->isSingular()) {
                    if (curVtx.mtrl->isTranslucent()) {
                        // cur頂点のひとつ前の頂点に基づいて、物体に入り込んでいるのか、それとも出て行くのかを判定する.
                        const vec3 intoCurVtxDir = normalize(curVtx.pos - prevVtx->pos);

                        // prevVtx から curVtx へのベクトルが、スペキュラ物体に入るのか、それとも出るのか.
                        const bool into = dot(intoCurVtxDir, curVtx.nml) < real(0);

                        const vec3 from_new_orienting_normal = into ? curVtx.nml : -curVtx.nml;

                        auto sampling = refraction::check(
                            curVtx.mtrl,
                            intoCurVtxDir,
                            curVtx.nml,
                            from_new_orienting_normal);

                        if (sampling.isIdealRefraction) {
                            // 屈折.
                            pdf = real(1);
                        }
                        else if (sampling.isRefraction) {
                            // 反射 or 屈折.
                            pdf = dot(from_new_orienting_normal, normalizedTo) > real(0)
                                ? sampling.probReflection                // 反射.
                                : real(1) - sampling.probReflection;    // 屈折.
                        }
                        else {
                            // 全反射.
                            pdf = real(1);
                        }
                    }
                    else {
                        pdf = real(1);
                    }
                }
                else {
                    pdf = curVtx.mtrl->pdf(curVtx.orienting_normal, wi, wo, curVtx.u, curVtx.v);
                }
            }
        }

        // 次の頂点の法線を、現在の頂点からの方向ベクトルに基づいて改めて求める.
        const vec3 next_new_orienting_normal = dot(to, nextVtx.nml) < 0.0
            ? nextVtx.nml
            : -nextVtx.nml;

        // 立体角測度に関する確率密度を面積測度に関する確率密度に変換.
        const real c = dot(-normalizedTo, next_new_orienting_normal);
        const real dist2 = squared_length(to);
        pdf *= c / dist2;

        return pdf;
    }

    real BDPT::computeMISWeight(
        camera* camera,
        real totalAreaPdf,
        const std::vector<Vertex>& eye_vs,
        int numEyeVtx,
        const std::vector<Vertex>& light_vs,
        int numLightVtx) const
    {
        // NOTE
        // https://www.slideshare.net/h013/edubpt-v100
        // p157 - p167

        // 光源上のサンプリング確率.
        const auto& beginLight = light_vs[0];
        const real areaPdf_y0 = beginLight.totalAreaPdf;

        // カメラ上のサンプリング確率.
        const auto& beginEye = eye_vs[0];
        const real areaPdf_x0 = beginEye.totalAreaPdf;

        std::vector<const Vertex*> vs(numEyeVtx + numLightVtx);

        // 頂点を一列に並べる。
        // vs[0] = y0, vs[1] = y1, ... vs[k-1] = x1, vs[k] = x0
        
        // lightサブパス.
        for (int i = 0; i < numLightVtx; ++i) {
            vs[i] = &light_vs[i];
        }

        // eyeサブパス.
        for (int i = numEyeVtx - 1; i >= 0; --i) {
            vs[numLightVtx + numEyeVtx - 1 - i] = &eye_vs[i];
        }

        // 終点のインデックス.
        const int k = numLightVtx + numEyeVtx - 1;

        // pi1/pi を計算.
        std::vector<real> pi1_pi(numLightVtx + numEyeVtx);
        {
            {
                const auto* vtx = vs[0];
                auto rr = russianRoulette(*vtx);
                auto areaPdf_y1 = computAreaPdf(camera, vs, 2, 1, 0);
                pi1_pi[0] = areaPdf_y0 / (areaPdf_y1 * rr);
            }

            // ロシアンルーレットの確率は打ち消しあうのでいらない？
            for (int i = 1; i < k; i++)
            {
                auto a = computAreaPdf(camera, vs, i - 2, i - 1, i);
                auto b = computAreaPdf(camera, vs, i + 2, i + 1, i);
                pi1_pi[i] = a / b;
            }

            {
                const auto* vtx = vs[k];
                auto rr = russianRoulette(*vtx);
                auto areaPdf_x1 = computAreaPdf(camera, vs, k - 2, k - 1, k);
                pi1_pi[k] = (areaPdf_x1 * rr) / areaPdf_x0;
            }
        }

        // pを求める
        std::vector<real> p(numEyeVtx + numLightVtx + 1);
        {
            // 真ん中にtotalAreaPdfをセット.
            p[numLightVtx] = totalAreaPdf;

            // 真ん中を起点に半分ずつ計算.

            // lightサブパス.
            for (int i = numLightVtx; i <= k; ++i) {
                p[i + 1] = p[i] * pi1_pi[i];
            }

            // eyeサブパス.
            for (int i = numLightVtx - 1; i >= 0; --i) {
                p[i] = p[i + 1] / pi1_pi[i];
            }

            for (int i = 0; i < vs.size(); ++i) {
                const auto& vtx = *vs[i];

                // 方向が一意に決まるので、影響を及ぼさない.
                if (vtx.mtrl && vtx.mtrl->isSingular()) {
                    p[i] = 0.0;
                    p[i + 1] = 0.0;
                }
            }
        }

        // Power-heuristic
        real misWeight = real(0);
        for (int i = 0; i < p.size(); ++i) {
            const real v = p[i] / p[numLightVtx];
            misWeight += v * v; // beta = 2
        }

        if (misWeight > real(0)) {
            misWeight = real(1) / misWeight;
        }

        return misWeight;
    }

    void BDPT::combine(
        const context& ctxt,
        int x, 
        int y,
        std::vector<Result>& result,
        const std::vector<Vertex>& eye_vs,
        const std::vector<Vertex>& light_vs,
        scene* scene,
        camera* camera) const
    {
        const int eyeNum = (int)eye_vs.size();
        const int lightNum = (int)light_vs.size();

        for (int numEyeVtx = 1; numEyeVtx <= eyeNum; ++numEyeVtx)
        {
            for (int numLightVtx = 1; numLightVtx <= lightNum; ++numLightVtx)
            {
                int targetX = x;
                int targetY = y;

                // それぞれのパスの端点.
                const Vertex& eye_end = eye_vs[numEyeVtx - 1];
                const Vertex& light_end = light_vs[numLightVtx - 1];

                // トータルの確率密度計算.
                const real totalAreaPdf = eye_end.totalAreaPdf * light_end.totalAreaPdf;
                if (totalAreaPdf == 0) {
                    // 何もなかったので、何もしない.
                    continue;
                }

                // MCスループット.
                vec3 eyeThroughput = eye_end.throughput;
                vec3 lightThroughput = light_end.throughput;

                // 頂点を接続することで新しく導入される項.
                vec3 throughput = vec3(1);

                // numLightVtx == 1のとき、非完全拡散光源の場合は相手の頂点の位置次第でMCスループットが変化するため改めて光源からの放射輝度値を計算する.
                if (numLightVtx == 1) {
                    // TODO
                    // 今回は完全拡散光源なので単純にemissionの値を入れる.
                    const auto& lightVtx = light_vs[0];
                    lightThroughput = lightVtx.light->getLe();
                }

                // 端点間が接続できるか.
                const vec3 lightEndToEyeEnd = eye_end.pos - light_end.pos;
                ray r(light_end.pos, normalize(lightEndToEyeEnd));

                hitrecord rec;
                Intersection isect;
                bool isHit = scene->hit(ctxt, r, AT_MATH_EPSILON, AT_MATH_INF, rec, isect);

                if (eye_end.objType == ObjectType::Lens) {
                    if (camera->isPinhole()) {
                        break;
                    }
                    else {
                        // lightサブパスを直接レンズにつなげる.
                        vec3 posOnLens;
                        vec3 posOnObjectplane;
                        vec3 posOnImagesensor;
                        int px, py;

                        const auto lens_t = camera->hitOnLens(
                            r,
                            posOnLens,
                            posOnObjectplane,
                            posOnImagesensor,
                            px, py);

                        if (AT_MATH_EPSILON < lens_t
                            && lens_t < isect.t)
                        {
                            // レイがレンズにヒット＆イメージセンサにヒット.
                            targetX = aten::clamp(px, 0, m_width - 1);
                            targetY = aten::clamp(py, 0, m_height - 1);

                            const real W_dash = camera->getWdash(
                                rec.p,
                                rec.normal,
                                posOnImagesensor,
                                posOnLens,
                                posOnObjectplane);

                            throughput *= W_dash;
                        }
                        else {
                            // lightサブパスを直接レンズにつなげようとしたが、遮蔽されたりイメージセンサにヒットしなかった場合、終わり.
                            continue;
                        }
                    }
                }
                else if (eye_end.objType == ObjectType::Light) {
                    // eyeサブパスの端点が光源（反射率0）だった場合は重みがゼロになりパス全体の寄与もゼロになるので、処理終わり.
                    // 光源は反射率0を想定している.
                    continue;
                }
                else {
                    if (eye_end.mtrl->isSingular()) {
                        // eyeサブパスの端点がスペキュラやだった場合は重みがゼロになりパス全体の寄与もゼロになるので、処理終わり.
                        // スペキュラの場合は反射方向で一意に決まり、lighサブパスの端点への方向が一致する確率がゼロ.
                        continue;
                    }
                    else {
                        // 端点同士が別の物体で遮蔽されるかどうかを判定する。遮蔽されていたら処理終わり.
                        const real len = length(eye_end.pos - rec.p);
                        if (len >= AT_MATH_EPSILON) {
                            continue;
                        }

                        const auto& bsdf = eye_end.bsdf;
                        throughput *= bsdf;
                    }
                }

                if (light_end.objType == ObjectType::Lens) {
                    // lightサブパスの端点がレンズだった場合は重みがゼロになりパス全体の寄与もゼロになるので、処理終わり.
                    // レンズ上はスペキュラとみなす.
                    continue;
                }
                else if (light_end.objType == ObjectType::Light) {
                    // 光源の反射率0を仮定しているため、ライトトレーシングの時、最初の頂点以外は光源上に頂点生成されない.
                    // num_light_vertex == 1以外でここに入ってくることは無い.
                }
                else {
                    if (light_end.mtrl->isSingular()) {
                        // eyeサブパスの端点がスペキュラやだった場合は重みがゼロになりパス全体の寄与もゼロになるので、処理終わり.
                        // スペキュラの場合は反射方向で一意に決まり、lighサブパスの端点への方向が一致する確率がゼロ.
                        continue;
                    }
                    else {
                        const auto& bsdf = light_end.bsdf;
                        throughput *= bsdf;
                    }
                }

                // 端点間のジオメトリファクタ
                {
                    real cx = dot(normalize(-lightEndToEyeEnd), eye_end.orienting_normal);
                    cx = std::max(cx, real(0));

                    real cy = dot(normalize(lightEndToEyeEnd), light_end.orienting_normal);
                    cy = std::max(cy, real(0));

                    const real dist2 = squared_length(lightEndToEyeEnd);

                    const real G = cx * cy / dist2;

                    throughput *= G;
                }

                // MIS.
                const real misWeight = computeMISWeight(
                    camera,
                    totalAreaPdf,
                    eye_vs, numEyeVtx,
                    light_vs, numLightVtx);

                if (misWeight <= real(0)) {
                    continue;
                }

                // 最終的なモンテカルロコントリビューション =
                //    MIS重み.
                //    * 頂点を接続することで新しく導入される項.
                //  * eyeサブパススループット.
                //    * lightサブパススループット.
                //    / パスのサンプリング確率密度の総計.
                const vec3 contrib = misWeight * throughput * eyeThroughput * lightThroughput / totalAreaPdf;
                result.push_back(Result(
                    contrib,
                    targetX, targetY,
                    numEyeVtx <= 1 ? false : true));
            }
        }
    }

    void BDPT::onRender(
        const context& ctxt,
        Destination& dst,
        scene* scene,
        camera* camera)
    {
        m_width = dst.width;
        m_height = dst.height;
        uint32_t samples = dst.sample;

        m_maxDepth = dst.maxDepth;

        const real divPixelProb = real(1) / (real)(m_width * m_height);

        // TODO
        /*
        m_rrDepth = dst.russianRouletteDepth;

        if (m_rrDepth > m_maxDepth) {
        m_rrDepth = m_maxDepth - 1;
        }
        */

        auto threadnum = OMPUtil::getThreadNum();

        std::vector<std::vector<vec4>> image(threadnum);

        for (int i = 0; i < threadnum; i++) {
            image[i].resize(m_width * m_height);
        }

#if defined(ENABLE_OMP) && !defined(BDPT_DEBUG)
#pragma omp parallel
#endif
        {
            auto idx = OMPUtil::getThreadIdx();

            auto time = timer::getSystemTime();

#if defined(ENABLE_OMP) && !defined(BDPT_DEBUG)
#pragma omp for
#endif
            for (int y = 0; y < m_height; y++) {
                for (int x = 0; x < m_width; x++) {
                    int pos = y * m_width + x;

                    for (uint32_t i = 0; i < samples; i++) {
                        auto scramble = aten::getRandom(pos) * 0x1fe3434f;

                        //XorShift rnd(scramble + time.milliSeconds);
                        //Halton rnd(scramble + time.milliSeconds);
                        //Sobol rnd(scramble + time.milliSeconds);
                        //WangHash rnd(scramble + time.milliSeconds);
                        CMJ rnd;
                        rnd.init(time.milliSeconds, i, scramble);

                        std::vector<Result> result;

                        std::vector<Vertex> eyevs;
                        std::vector<Vertex> lightvs;

                        auto eyeRes = genEyePath(ctxt, eyevs, x, y, &rnd, scene, camera);
                        
#if 0
                        if (eyeRes.isTerminate) {
                            int pos = eyeRes.y * m_width + eyeRes.x;
                            image[idx][pos] += vec4(eyeRes.contrib, 1);
                        }
#else
                        auto lightNum = scene->lightNum();
                        for (uint32_t n = 0; n < lightNum; n++) {
                            auto light = scene->getLight(n);
                            auto lightRes = genLightPath(ctxt, lightvs, light, &rnd, scene, camera);

                            if (eyeRes.isTerminate) {
                                const real misWeight = computeMISWeight(
                                    camera,
                                    eyevs[eyevs.size() - 1].totalAreaPdf,
                                    eyevs,
                                    (const int)eyevs.size(),   // num_eye_vertex
                                    lightvs,
                                    0);                         // num_light_vertex

                                const vec3 contrib = misWeight * eyeRes.contrib;
                                result.push_back(Result(contrib, eyeRes.x, eyeRes.y, true));
                            }

                            if (lightRes.isTerminate) {
                                const real misWeight = computeMISWeight(
                                    camera,
                                    lightvs[lightvs.size() - 1].totalAreaPdf,
                                    eyevs,
                                    0,                            // num_eye_vertex
                                    lightvs,
                                    (const int)lightvs.size());    // num_light_vertex

                                const vec3 contrib = misWeight * lightRes.contrib;
                                result.push_back(Result(contrib, lightRes.x, lightRes.y, false));
                            }

                            combine(
                                ctxt, 
                                x, y,
                                result, 
                                eyevs,
                                lightvs,
                                scene,
                                camera);

#if 1
                            for (int i = 0; i < (int)result.size(); i++) {
                                const auto& res = result[i];

                                // TODO
                                // FIXME
                                // I have to research why contribute value is invalid.
                                if (isInvalidColor(res.contrib)) {
                                    //AT_PRINTF("Invalid(%d/%d[%d])\n", x, y, i);
                                    continue;
                                }

                                const int pos = res.y * m_width + res.x;

                                if (res.isStartFromPixel) {
                                    image[idx][pos] += vec4(res.contrib, 1);
                                }
                                else {
                                    // 得られたサンプルについて、サンプルが現在の画素（x,y)から発射されたeyeサブパスを含むものだった場合
                                    // Ixy のモンテカルロ推定値はsamples[i].valueそのものなので、そのまま足す。その後、下の画像出力時に発射された回数の総計（iteration_per_thread * num_threads)で割る.
                                    //
                                    // 得られたサンプルについて、現在の画素から発射されたeyeサブパスを含むものではなかった場合（lightサブパスが別の画素(x',y')に到達した場合）は
                                    // Ix'y' のモンテカルロ推定値を新しく得たわけだが、この場合、画像全体に対して光源側からサンプルを生成し、たまたまx'y'にヒットしたと考えるため
                                    // このようなサンプルについては最終的に光源から発射した回数の総計で割って、画素への寄与とする必要がある.
                                    image[idx][pos] += vec4(res.contrib * divPixelProb, 1);
                                }
                            }
#endif

                        }
#endif
                    }
                }
            }
        }

        std::vector<vec4> tmp(m_width * m_height);

        for (int i = 0; i < threadnum; i++) {
            auto& img = image[i];
            for (int y = 0; y < m_height; y++) {
                for (int x = 0; x < m_width; x++) {
                    int pos = y * m_width + x;

                    auto clr = img[pos] / samples;
                    clr.w = 1;

                    tmp[pos] += clr;
                }
            }
        }

#if defined(ENABLE_OMP) && !defined(BDPT_DEBUG)
#pragma omp parallel for
#endif
        for (int y = 0; y < m_height; y++) {
            for (int x = 0; x < m_width; x++) {
                int pos = y * m_width + x;

                auto clr = tmp[pos];
                clr.w = 1;

                dst.buffer->put(x, y, clr);
            }
        }
    }
}