#pragma once

#include "renderer/renderer.h"
#include "scene/scene.h"
#include "camera/camera.h"
#include "sampler/sampler.h"

namespace aten
{
    class BDPT : public Renderer {
    public:
        BDPT() {}
        ~BDPT() {}

    public:
        virtual void onRender(
            const context& ctxt,
            Destination& dst,
            scene* scene,
            camera* camera) override;

    private:
        enum ObjectType {
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

            const hitable* obj{ nullptr };
            const material* mtrl{ nullptr };

            aten::Light* light{ nullptr };

            Vertex(
                const vec3& p,
                const vec3& _nml,
                const vec3& _orienting_normal,
                ObjectType type,
                real _totalAreapdf,
                const vec3& th,
                const vec3& _bsdf,
                const hitable* _obj,
                const material* _mtrl,
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
                aten::Light* _light)
                : pos(p), nml(_nml), orienting_normal(_orienting_normal), objType(type), totalAreaPdf(_totalAreapdf), throughput(th), 
                bsdf(_bsdf), light(_light)
            {}
        };

        struct Result {
            vec3 contrib;
            int x;
            int y;

            union {
                bool isTerminate;
                bool isStartFromPixel;
            };

            Result(vec3 c, int _x, int _y, bool _isTerminate)
                : contrib(c), x(_x), y(_y), isTerminate(_isTerminate)
            {}
        };

        Result genEyePath(
            const context& ctxt,
            std::vector<Vertex>& vs,
            int x, int y,
            sampler* sampler,
            scene* scene,
            camera* camera) const;

        Result genLightPath(
            const context& ctxt,
            std::vector<Vertex>& vs,
            aten::Light* light,
            sampler* sampler,
            scene* scene,
            camera* camera) const;

        real computAreaPdf(
            camera* camera,
            const std::vector<const Vertex*>& vs,
            const int prev_from_idx,
            const int from_idx,
            const int next_idx) const;

        real computeMISWeight(
            camera* camera,
            real totalAreaPdf,
            const std::vector<Vertex>& eye_vs,
            int numEyeVtx,
            const std::vector<Vertex>& light_vs,
            int numLightVtx) const;

        void combine(
            const context& ctxt,
            int x, 
            int y,
            std::vector<Result>& result,
            const std::vector<Vertex>& eye_vs,
            const std::vector<Vertex>& light_vs,
            scene* scene,
            camera* camera) const;

        static inline real russianRoulette(const Vertex& vtx);

    private:
        int m_maxDepth{ 1 };

        int m_width;
        int m_height;
    };
}
