#pragma once

#include "math/vec3.h"

namespace aten::sky {
    struct DensityProfile {
        float width;
        float inv_height_scale;
        float exp_term{ 1.0F };
    };

    static inline AT_DEVICE_API float GetLayerDensity(
        const DensityProfile& layer,
        const float altitude)
    {
        // オゾン層、Rayleigh散乱層、Mie散乱層などの大気の層の密度分布を表すための構造体.
        // Rayleigh散乱層、Mie散乱層では、exp_termが1で、exp_scaleが exp(-h/H) の形での -1/H となる.
        // それ以外の、変数はオゾン層で利用される (demo.cc L242).
        float density = layer.exp_term * exp(altitude * layer.inv_height_scale);
        return aten::clamp(density, 0.0F, 1.0F);
    }

    static inline AT_DEVICE_API float GetProfileDensity(
        const DensityProfile& profile,
        const float altitude)
    {
        // 散乱層が複数層になっているか
        // 基本的には、オゾン層のみが複数層になる.
        return altitude < profile.width
            ? GetLayerDensity(profile, altitude)
            : 0.0F;
    }

    /*
    The atmosphere parameters are then defined by the following struct:
    */

    struct AtmosphereParameters {
        // The solar irradiance at the top of the atmosphere.
        aten::vec3 solar_irradiance;

        // The sun's angular radius. Warning: the implementation uses approximations
        // that are valid only if this angle is smaller than 0.1 radians.

        // https://physmemo.shakunage.net/phys/diameter/example1.html
        // 太陽の視半径.
        float sun_angular_radius;

        // The distance between the planet center and the bottom of the atmosphere.
        // 地球の半径.
        float bottom_radius;

        // The distance between the planet center and the top of the atmosphere.
        // 大気境界までの半径.
        float top_radius;

        // The density profile of air molecules, i.e. a function from altitude to
        // dimensionless values between 0 (null density) and 1 (maximum density).
        DensityProfile rayleigh_density;

        // The scattering coefficient of air molecules at the altitude where their
        // density is maximum (usually the bottom of the atmosphere), as a function of
        // wavelength. The scattering coefficient at altitude h is equal to
        // 'rayleigh_scattering' times 'rayleigh_density' at this altitude.
        aten::vec3 rayleigh_scattering;

        // The density profile of aerosols, i.e. a function from altitude to
        // dimensionless values between 0 (null density) and 1 (maximum density).
        DensityProfile mie_density;

        // The scattering coefficient of aerosols at the altitude where their density
        // is maximum (usually the bottom of the atmosphere), as a function of
        // wavelength. The scattering coefficient at altitude h is equal to
        // 'mie_scattering' times 'mie_density' at this altitude.
        aten::vec3 mie_scattering;

        // The extinction coefficient of aerosols at the altitude where their density
        // is maximum (usually the bottom of the atmosphere), as a function of
        // wavelength. The extinction coefficient at altitude h is equal to
        // 'mie_extinction' times 'mie_density' at this altitude.
        aten::vec3 mie_extinction;

        // The asymmetry parameter for the Cornette-Shanks phase function for the
        // aerosols.
        float mie_phase_function_g;


        // The density profile of air molecules that absorb light (e.g. ozone), i.e.
        // a function from altitude to dimensionless values between 0 (null density)
        // and 1 (maximum density).

        // TODO
        //DensityProfile absorption_density;

        // The extinction coefficient of molecules that absorb light (e.g. ozone) at
        // the altitude where their density is maximum, as a function of wavelength.
        // The extinction coefficient at altitude h is equal to
        // 'absorption_extinction' times 'absorption_density' at this altitude.

        // TODO
        //aten::vec3 absorption_extinction;

        // The average albedo of the ground.
        aten::vec3 ground_albedo;

        // The cosine of the maximum Sun zenith angle for which atmospheric scattering
        // must be precomputed (for maximum precision, use the smallest Sun zenith
        // angle yielding negligible sky light radiance values. For instance, for the
        // Earth case, 102 degrees is a good choice - yielding mu_s_min = -0.2).

        // https://gemini.google.com/share/33ef5a33da65
        // 太陽天頂角（Sun Zenith Angle）」とは、天頂角（Zenith Angle）は「真上（天頂）」を 0度 とした角度のこと.
        //  - 0度: 太陽が真上にある（正午）/
        //  - 90度: 太陽が地平線にある（日の出・日の入り）.
        //  - 90度 以上: 太陽が地平線より下にある（夜）.
        // なぜ「102度」なのか？
        // 太陽が地平線（90度）に沈んでも、すぐには真っ暗にならず、上空の大気に光が当たって散乱するため、
        // しばらくは「薄明（トワイライト）」として空が明るい.
        // しかし、太陽がある程度深く沈むと、散乱光は極めて弱くなり、無視できるレベル（negligible）になる.
        // 例では、地球の場合 102度 まで沈めば、それ以上計算しても結果はほぼ変わらない（真っ暗とみなせる）としている.
        // 102度という数字には、天文学的・気象学的な明確な根拠があり、「航海薄明（Nautical Twilight）」と「天文薄明（Astronomical Twilight）」の境界付近を指している.
        // μ_s = cos(102度) ≈ -0.2 となる.
        float mu_s_min;
    };
}
