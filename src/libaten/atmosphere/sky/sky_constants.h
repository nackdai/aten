#pragma once

#include "atmosphere/sky/unit_quantity.h"

#include "math/math.h"

namespace aten::sky {
    // As [mm].
    constexpr int32_t LambdaMin = 360;
    constexpr int32_t LambdaMax = 830;

    constexpr float LambdaR = 680.0;
    constexpr float LambdaG = 550.0;
    constexpr float LambdaB = 440.0;

    // Wavelength independent solar irradiance "spectrum" (not physically
    // realistic, but was used in the original implementation).
    // // Values in W.m^-2 per nanometer. -> W.m^-2.nm^-1
    constexpr float ConstantSolarIrradiance = 1.5F;

    constexpr Length BottomRadius = 6360.0_km;  // Rg = 6360 km in meter.
    constexpr Length TopRadius = 6420.0_km;     // Rt = 6420 km in meter.

    constexpr float Rayleigh = 1.24062e-6F;
    constexpr Length RayleighScaleHeight = 8.0_km; // HR = 8 km in meter.

    constexpr Length MieScaleHeight = 1.2_km;  // HM = 1.2 km in meter.

    // https://gemini.google.com/share/269fb9c1472b
    constexpr float MieAngstromAlpha = 0.0F;            // No unit
    constexpr float MieAngstromBeta = 5.328e-3F;        // No unit
    constexpr float MieSingleScatteringAlbedo = 0.9F;   // No unit
    constexpr float MiePhaseFunctionG = 0.8F;

    constexpr float GroundAlbedo = 0.1F;

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
    constexpr float MaxSunZenithAngle = Deg2Rad(102.0F);

    // https://physmemo.shakunage.net/phys/diameter/diameter.html
    // 太陽の視直径の平均視直径 0.5331 度 を ラジアンに変換した値.
    // 半径なので、1/2 する.
    constexpr double SunAngularRadius = 0.00935 / 2.0;

    // 論文内の 6. Implementation, results and discussion より:

    // We store T(r,µ) and E(r,µ) in 64x256 and 16x64 textures.
    // For T(r,µ) 64x256
    // TRANSMITTANCE_TEXTURE_WIDTH: 256
    // TRANSMITTANCE_TEXTURE_HEIGHT: 64

    constexpr int TRANSMITTANCE_TEXTURE_WIDTH = 256;
    constexpr int TRANSMITTANCE_TEXTURE_HEIGHT = 64;

    // We store S(ur,uµ,uµs,uν) =[C∗,CM,r] in a 32×128×32×8 table,
    // seen as 8 3D tables packed in a single 32 × 128 × 256 RGBA texture.

    // SCATTERING_TEXTURE_WIDTH: 256 = SCATTERING_TEXTURE_NU_SIZE * SCATTERING_TEXTURE_MU_S_SIZE = 32 x 8
    // SCATTERING_TEXTURE_HEIGHT: 128 = SCATTERING_TEXTURE_MU_SIZE
    // SCATTERING_TEXTURE_DEPTH: 32 = SCATTERING_TEXTURE_R_SIZE

    constexpr int SCATTERING_TEXTURE_R_SIZE = 32;
    constexpr int SCATTERING_TEXTURE_MU_SIZE = 128;
    constexpr int SCATTERING_TEXTURE_MU_S_SIZE = 32;
    constexpr int SCATTERING_TEXTURE_NU_SIZE = 8;

    constexpr int SCATTERING_TEXTURE_WIDTH =
        SCATTERING_TEXTURE_NU_SIZE * SCATTERING_TEXTURE_MU_S_SIZE;
    constexpr int SCATTERING_TEXTURE_HEIGHT = SCATTERING_TEXTURE_MU_SIZE;
    constexpr int SCATTERING_TEXTURE_DEPTH = SCATTERING_TEXTURE_R_SIZE;

    // We store T(r,µ) and E(r,µ) in 64x256 and 16x64 textures.
    // For E(r,µ), 16x64 textures.
    // IRRADIANCE_TEXTURE_WIDTH: 64
    // IRRADIANCE_TEXTURE_HEIGHT: 16

    constexpr int IRRADIANCE_TEXTURE_WIDTH = 64;
    constexpr int IRRADIANCE_TEXTURE_HEIGHT = 16;
}
