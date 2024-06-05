#include "scene/scene.h"
#include "misc/color.h"
#include "geometry/transformable.h"
#include "light/light_impl.h"

namespace aten {
    void scene::render(
        aten::hitable::FuncPreDraw func,
        std::function<bool(const std::shared_ptr<aten::hitable>&)> funcIfDraw,
        const context& ctxt) const
    {
        uint32_t triOffset = 0;

        for (auto h : m_list) {
            bool willDraw = funcIfDraw ? funcIfDraw(h) : true;

            if (willDraw) {
                h->render(func, ctxt, aten::mat4::Identity, aten::mat4::Identity, -1, triOffset);
            }

            auto item = h->getHasObject();
            if (item) {
                triOffset += item->getTriangleCount();
            }
        }
    }
}
