#pragma once

#include <vector>
#include "light/light.h"
#include "renderer/envmap.h"

namespace aten {
	class ImageBasedLight : public Light {
	public:
		ImageBasedLight()
			: Light(LightTypeIBL)
		{}
		ImageBasedLight(envmap* envmap)
			: Light(LightTypeIBL)
		{
			setEnvMap(envmap);
		}

		ImageBasedLight(Values& val)
			: Light(LightTypeIBL, val)
		{
			texture* tex = (texture*)val.get("envmap", nullptr);
			
			envmap* bg = new envmap();
			bg->init(tex);

			setEnvMap(bg);
		}

		virtual ~ImageBasedLight() {}

	public:
		void setEnvMap(envmap* envmap)
		{
			if (m_param.envmap.ptr != envmap) {
				m_param.envmap.ptr = envmap;
				preCompute();
			}
		}

		const envmap* getEnvMap() const
		{
			return (envmap*)m_param.envmap.ptr;
		}

		real samplePdf(const ray& r) const;

		virtual LightSampleResult sample(const vec3& org, sampler* sampler) const override final;

	private:
		void preCompute();

	private:
		real m_avgIllum{ real(0) };

		// v•ûŒü‚Ìcdf(cumulative distribution function = —İÏ•ª•zŠÖ” = sum of pdf).
		std::vector<real> m_cdfV;

		// u•ûŒü‚Ìcdf(cumulative distribution function = —İÏ•ª•zŠÖ” = sum of pdf).
		// —ñ‚²‚Æ‚É•Û‚·‚é.
		std::vector<std::vector<real>> m_cdfU;
	};
}