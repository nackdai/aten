#pragma once

#include "material/material.h"
#include "texture/texture.h"

namespace aten
{
	class DisneyBRDF : public material {
	public:
		struct Parameter {
			vec3 baseColor{ vec3(real(0.82), real(0.67), real(0.16)) };
			real metallic{ 0 };
			real subsurface{ 0 };
			real specular{ 0.5 };
			real roughness{ 0.5 };
			real specularTint{ 0 };
			real anisotropic{ 0 };
			real sheen{ 0 };
			real sheenTint{ 0.5 };
			real clearcoat{ 0 };
			real clearcoatGloss{ 1 };
		};
	public:
		DisneyBRDF() {}
		DisneyBRDF(
			vec3 baseColor,
			real subsurface,
			real metallic,
			real specular,
			real specularTint,
			real roughness,
			real anisotropic,
			real sheen,
			real sheenTint,
			real clearcoat,
			real clearcoatGloss,
			texture* albedoMap = nullptr,
			texture* normalMap = nullptr,
			texture* roughnessMap = nullptr)
			: material(baseColor, 1, albedoMap, normalMap)
		{
			m_param.baseColor = baseColor;
			m_param.subsurface = aten::clamp<real>(subsurface, 0, 1);
			m_param.metallic = aten::clamp<real>(metallic, 0, 1);
			m_param.specular = aten::clamp<real>(specular, 0, 1);
			m_param.specularTint = aten::clamp<real>(specularTint, 0, 1);
			m_param.roughness = aten::clamp<real>(roughness, 0, 1);
			m_param.anisotropic = aten::clamp<real>(anisotropic, 0, 1);
			m_param.sheen = aten::clamp<real>(sheen, 0, 1);
			m_param.sheenTint = aten::clamp<real>(sheenTint, 0, 1);
			m_param.clearcoat = aten::clamp<real>(clearcoat, 0, 1);
			m_param.clearcoatGloss = aten::clamp<real>(clearcoatGloss, 0, 1);

			m_param.roughnessMap.tex = roughnessMap;
		}

		DisneyBRDF(
			const Parameter& param,
			texture* albedoMap = nullptr,
			texture* normalMap = nullptr,
			texture* roughnessMap = nullptr)
			: material(param.baseColor, 1, albedoMap, normalMap)
		{
			m_param.baseColor = param.baseColor;
			m_param.subsurface = aten::clamp<real>(param.subsurface, 0, 1);
			m_param.metallic = aten::clamp<real>(param.metallic, 0, 1);
			m_param.specular = aten::clamp<real>(param.specular, 0, 1);
			m_param.specularTint = aten::clamp<real>(param.specularTint, 0, 1);
			m_param.roughness = aten::clamp<real>(param.roughness, 0, 1);
			m_param.anisotropic = aten::clamp<real>(param.anisotropic, 0, 1);
			m_param.sheen = aten::clamp<real>(param.sheen, 0, 1);
			m_param.sheenTint = aten::clamp<real>(param.sheenTint, 0, 1);
			m_param.clearcoat = aten::clamp<real>(param.clearcoat, 0, 1);
			m_param.clearcoatGloss = aten::clamp<real>(param.clearcoatGloss, 0, 1);

			m_param.roughnessMap.tex = roughnessMap;
		}

		DisneyBRDF(Values& val)
			: material(val)
		{
			// TODO
			// Clamp parameters.
			m_param.subsurface = val.get("subsurface", m_param.subsurface);
			m_param.metallic = val.get("metallic", m_param.metallic);
			m_param.specular = val.get("specular", m_param.specular);
			m_param.specularTint = val.get("specularTint", m_param.specularTint);
			m_param.roughness = val.get("roughness", m_param.roughness);
			m_param.anisotropic = val.get("anisotropic", m_param.anisotropic);
			m_param.sheen = val.get("sheen", m_param.sheen);
			m_param.sheenTint = val.get("sheenTint", m_param.sheenTint);
			m_param.clearcoat = val.get("clearcoat", m_param.clearcoat);
			m_param.clearcoatGloss = val.get("clearcoatGloss", m_param.clearcoatGloss);
			m_param.roughnessMap = val.get("roughnessmap", m_param.roughnessMap);
		}

		virtual ~DisneyBRDF() {}

	public:
		virtual bool isGlossy() const override final
		{
			return (m_param.roughness == 1 ? false : true);
		}

		virtual real pdf(
			const vec3& normal, 
			const vec3& wi,
			const vec3& wo,
			real u, real v) const override final;

		virtual vec3 sampleDirection(
			const ray& ray,
			const vec3& normal,
			real u, real v,
			sampler* sampler) const override final;

		virtual vec3 bsdf(
			const vec3& normal, 
			const vec3& wi,
			const vec3& wo,
			real u, real v) const override final;

		virtual MaterialSampling sample(
			const ray& ray,
			const vec3& normal,
			const hitrecord& hitrec,
			sampler* sampler,
			real u, real v,
			bool isLightPath = false) const override final;

	private:
		real pdf(
			const vec3& V,
			const vec3& N,
			const vec3& L,
			const vec3& X,
			const vec3& Y,
			real u, real v) const;

		vec3 sampleDirection(
			const vec3& V,
			const vec3& N,
			const vec3& X,
			const vec3& Y,
			real u, real v,
			sampler* sampler) const;

		vec3 bsdf(
			real& fresnel,
			const vec3& V,
			const vec3& N,
			const vec3& L,
			const vec3& X,
			const vec3& Y,
			real u, real v) const;
	};
}
