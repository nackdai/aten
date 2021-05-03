#pragma once

#include "visualizer/visualizer.h"
#include "visualizer/blitter.h"

namespace aten {
    class GeometryRendering : public visualizer::PreProc{
    public:
        GeometryRendering() = default;
        virtual ~GeometryRendering() = default;

    public:
        virtual void operator()(
            const vec4* src,
            uint32_t width, uint32_t height,
            vec4* dst) override final;

        void setParam(
            uint32_t ratio,
            vec4* direct,
            vec4* indirect,
            vec4* idx)
        {
            m_ratio = ratio;
            m_direct = direct;
            m_indirect = indirect;
            m_idx = idx;
        }

    private:
        struct Pos {
            int x;
            int y;
        };

        union Idx {
            struct {
                uint32_t shapeid;
                uint32_t mtrlid;
            };
            uint64_t id;
        };

        static inline void getIdx(Idx& idx, const vec4& v);

    private:
        uint32_t m_ratio{ 1 };
        vec4* m_direct{ nullptr };
        vec4* m_indirect{ nullptr };
        vec4* m_idx{ nullptr };
    };
}
