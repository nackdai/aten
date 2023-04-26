#pragma once

#include "renderer/renderer.h"
#include "scene/scene.h"
#include "camera/camera.h"
#include "sampler/sampler.h"

namespace aten
{
    class BDPT : public Renderer {
    public:
        BDPT() = default;
        ~BDPT() = default;

    public:
        virtual void onRender(
            const context& ctxt,
            Destination& dst,
            scene* scene,
            camera* camera) override;

    private:
        enum class ObjectType {
            Light,
            Lens,
            Object,
        };

        struct Vertex {
            vec3 pos;
            vec3 nml;
            vec3 orienting_normal;

            ObjectType objType;

            real totalAreaPdf;
            vec3 throughput{ real(1) };

            vec3 bsdf{ real(0) };

            real u{ real(0) };
            real v{ real(0) };

            std::shared_ptr<const hitable> obj;
            std::shared_ptr<const material> mtrl;

            std::shared_ptr<const aten::Light> light;

            Vertex(
                const vec3& p,
                const vec3& _nml,
                const vec3& _orienting_normal,
                ObjectType type,
                real _totalAreapdf,
                const vec3& th,
                const vec3& _bsdf,
                const std::shared_ptr<const hitable>& _obj,
                const std::shared_ptr<const material>& _mtrl,
                real _u, real _v)
                : pos(p), nml(_nml), orienting_normal(_orienting_normal), objType(type), totalAreaPdf(_totalAreapdf), throughput(th),
                bsdf(_bsdf), obj(_obj), mtrl(_mtrl), u(_u), v(_v)
            {}

            Vertex(
                const vec3& p,
                const vec3& _nml,
                const vec3& _orienting_normal,
                ObjectType type,
                real _totalAreapdf,
                const vec3& th,
                const vec3& _bsdf,
                const std::shared_ptr<const aten::Light>& _light)
                : pos(p), nml(_nml), orienting_normal(_orienting_normal), objType(type), totalAreaPdf(_totalAreapdf), throughput(th),
                bsdf(_bsdf), light(_light)
            {}
        };

        struct Result {
            vec3 contrib;
            int32_t x;
            int32_t y;

            union {
                bool isTerminate;
                bool isStartFromPixel;
            };

            Result(vec3 c, int32_t _x, int32_t _y, bool _isTerminate)
                : contrib(c), x(_x), y(_y), isTerminate(_isTerminate)
            {}
        };

        Result genEyePath(
            const context& ctxt,
            std::vector<Vertex>& vs,
            int32_t x, int32_t y,
            sampler* sampler,
            scene* scene,
            camera* camera) const;

        Result genLightPath(
            const context& ctxt,
            std::vector<Vertex>& vs,
            const std::shared_ptr<aten::Light> light,
            sampler* sampler,
            scene* scene,
            camera* camera) const;

        real computAreaPdf(
            camera* camera,
            const std::vector<const Vertex*>& vs,
            const int32_t prev_from_idx,
            const int32_t from_idx,
            const int32_t next_idx) const;

        real computeMISWeight(
            camera* camera,
            real totalAreaPdf,
            const std::vector<Vertex>& eye_vs,
            int32_t numEyeVtx,
            const std::vector<Vertex>& light_vs,
            int32_t numLightVtx) const;

        void combine(
            const context& ctxt,
            int32_t x,
            int32_t y,
            std::vector<Result>& result,
            const std::vector<Vertex>& eye_vs,
            const std::vector<Vertex>& light_vs,
            scene* scene,
            camera* camera) const;

        static inline real russianRoulette(const Vertex& vtx);

    private:
        int32_t m_maxDepth{ 1 };

        int32_t m_width{ 0 };
        int32_t m_height{ 0 };
    };
}
