#include "scene/context.h"

namespace aten
{
    void context::build()
    {
        if (!m_vertices.empty()
            && !m_vb.isInitialized())
        {
            m_vb.init(
                sizeof(vertex),
                m_vertices.size(),
                0,
                &m_vertices[0]);
        }
    }

    int context::findMaterialIdxByName(const char* name) const
    {
        std::string strname(name);
        
        auto& materials = m_materials.getRealDataList();

        auto found = std::find_if(
            materials.begin(), materials.end(),
            [&](const material* mtrl) {
            return mtrl->nameString() == strname;
        });

        if (found != materials.end()) {
            auto* mtrl = *found;
            return mtrl->id();
        }

        return -1;
    }
}
