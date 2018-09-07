#include <atomic>
#include "material/material.h"
#include "light/light.h"

namespace AT_NAME
{
	std::vector<material*> material::g_materials;
	std::vector<const char*> material::g_mtrlTypeNames;

	static const char* mtrlTypeNames[] = {
		"emissive",
		"lambert",
		"ornenayar",
		"specular",
		"refraction",
		"blinn",
		"ggx",
		"beckman",
		"velvet",
		"disney_brdf",
		"carpaint",
		"toon",
		"layer",
	};
	AT_STATICASSERT(AT_COUNTOF(mtrlTypeNames) == (int)aten::MaterialType::MaterialTypeMax);

	uint32_t material::getMaterialNum()
	{
		return (uint32_t)g_materials.size();
	}

	material* material::getMaterial(uint32_t idx)
	{
		if (idx < g_materials.size()) {
			return g_materials[idx];
		}
		return nullptr;
	}

	bool material::deleteMaterial(material* mtrl, bool needDelete/*= false*/)
	{
		auto found = std::find(g_materials.begin(), g_materials.end(), mtrl);
		if (found != g_materials.end()) {
			g_materials.erase(found);

			// IDの振り直し...
			for (int i = 0; i < g_materials.size(); i++) {
				g_materials[i]->m_id = i;
			}

			if (needDelete) {
				delete mtrl;
			}

			return true;
		}

		return false;
	}

	void material::clearMaterialList()
	{
		g_materials.clear();
	}

	int material::findMaterialIdx(material* mtrl)
	{
		auto found = std::find(g_materials.begin(), g_materials.end(), mtrl);
		if (found != g_materials.end()) {
			auto id = std::distance(g_materials.begin(), found);
			AT_ASSERT(mtrl == g_materials[id]);
			return id;
		}
		return -1;
	}

	int material::findMaterialIdxByName(const char* name)
	{
		std::string strname(name);

		auto found = std::find_if(
			g_materials.begin(), g_materials.end(),
			[&](const material* mtrl) {
			return mtrl->nameString() == strname;
		});

		if (found != g_materials.end()) {
			const auto* mtrl = *found;
			return mtrl->m_id;
		}

		return -1;
	}

	const std::vector<material*>& material::getMaterials()
	{
		return g_materials;
	}

	const char* material::getMaterialTypeName(aten::MaterialType type)
	{
		initMaterialTypeName();
		return g_mtrlTypeNames[type];
	}

	std::vector<const char*>& material::getMaterialTypeName()
	{
		initMaterialTypeName();
		return g_mtrlTypeNames;
	}

	void material::initMaterialTypeName()
	{
		if (g_mtrlTypeNames.empty()) {
			for (auto name : mtrlTypeNames) {
				g_mtrlTypeNames.push_back(name);
			}
		}
	}

	int material::initMaterial(material* mtrl, bool local)
	{
		initMaterialTypeName();

		int id = -1;
		if (!local) {
			id = g_materials.size();
			g_materials.push_back(mtrl);
		}

		return id;
	}

	material::material(
		aten::MaterialType type, 
		const aten::MaterialAttribute& attrib,
		bool local/*= false*/)
		: m_param(type, attrib)
	{
		m_id = initMaterial(this, local);
	}

	material::material(
		aten::MaterialType type,
		const aten::MaterialAttribute& attrib,
		const aten::vec3& clr,
		real ior/*= 1*/,
		aten::texture* albedoMap/*= nullptr*/,
		aten::texture* normalMap/*= nullptr*/,
		bool local/*= false*/)
		: m_param(type, attrib)
	{
		m_id = initMaterial(this, local);

		m_param.baseColor = clr;
		m_param.ior = ior;

		setTextures(albedoMap, normalMap, nullptr);
	}

	material::material(
		aten::MaterialType type, 
		const aten::MaterialAttribute& attrib, 
		aten::Values& val,
		bool local/*= false*/)
		: m_param(type, attrib)
	{
		m_id = initMaterial(this, local);

		m_param.baseColor = val.get("baseColor", m_param.baseColor);
		m_param.ior = val.get("ior", m_param.ior);
		
		auto albedoMap = (aten::texture*)val.get<void*>("albedoMap", nullptr);
		auto normalMap = (aten::texture*)val.get<void*>("normalMap", nullptr);

		setTextures(albedoMap, normalMap, nullptr);
	}

	void material::setTextures(
		aten::texture* albedoMap,
		aten::texture* normalMap,
		aten::texture* roughnessMap)
	{
		m_param.albedoMap = albedoMap ? albedoMap->id() : -1;
		m_param.normalMap = normalMap ? normalMap->id() : -1;
		m_param.roughnessMap = roughnessMap ? roughnessMap->id() : -1;
	}

	material::~material()
	{
		deleteMaterial(this);
	}

	NPRMaterial::NPRMaterial(
		aten::MaterialType type,
		const aten::vec3& e, AT_NAME::Light* light)
		: material(type, MaterialAttributeNPR, e)
	{
		setTargetLight(light);
	}

	void NPRMaterial::setTargetLight(AT_NAME::Light* light)
	{
		m_targetLight = light;
	}

	const AT_NAME::Light* NPRMaterial::getTargetLight() const
	{
		return m_targetLight;
	}

	// NOTE
	// Schlick によるフレネル反射率の近似.
	// http://yokotakenji.me/log/math/4501/
	// https://en.wikipedia.org/wiki/Schlick%27s_approximation

	// NOTE
	// フレネル反射率について.
	// http://d.hatena.ne.jp/hanecci/20130525/p3

	real schlick(
		const aten::vec3& in,
		const aten::vec3& normal,
		real ni, real nt)
	{
		// NOTE
		// Fschlick(v,h) ≒ R0 + (1 - R0)(1 - cosΘ)^5
		// R0 = ((n1 - n2) / (n1 + n2))^2

		auto r0 = (ni - nt) / (ni + nt);
		r0 = r0 * r0;

		auto c = dot(in, normal);

		return r0 + (1 - r0) * aten::pow((1 - c), 5);
	}

	real computFresnel(
		const aten::vec3& in,
		const aten::vec3& normal,
		real ni, real nt)
	{
		real cos_i = dot(in, normal);

		bool isEnter = (cos_i > real(0));

		aten::vec3 n = normal;

		if (isEnter) {
			// レイが出ていくので、全部反対.
			auto tmp = nt;
			nt = real(1);
			ni = tmp;

			n = -n;
		}

		auto eta = ni / nt;

		auto sini2 = 1.f - cos_i * cos_i;
		auto sint2 = eta * eta * sini2;

		auto fresnel = schlick(
			in, 
			n, ni, nt);

		return fresnel;
	}

	AT_DEVICE_MTRL_API void material::applyNormalMap(
		const aten::vec3& orgNml,
		aten::vec3& newNml,
		real u, real v) const
	{
		applyNormalMap(m_param.normalMap, orgNml, newNml, u, v);
	}	
}
