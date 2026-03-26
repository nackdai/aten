#pragma once

#include <array>

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
    constexpr float SunAngularRadius = 0.00935F / 2.0F;

    // 論文内の 6. Implementation, results and discussion より:

    // We store T(r,µ) and E(r,µ) in 64x256 and 16x64 textures.
    // For T(r,µ) 64x256
    // TRANSMITTANCE_TEXTURE_WIDTH: 256
    // TRANSMITTANCE_TEXTURE_HEIGHT: 64

    constexpr int32_t TRANSMITTANCE_TEXTURE_WIDTH = 256;
    constexpr int32_t TRANSMITTANCE_TEXTURE_HEIGHT = 64;

    // We store S(ur,uµ,uµs,uν) =[C∗,CM,r] in a 32×128×32×8 table,
    // seen as 8 3D tables packed in a single 32 × 128 × 256 RGBA texture.

    // SCATTERING_TEXTURE_WIDTH: 256 = SCATTERING_TEXTURE_NU_SIZE * SCATTERING_TEXTURE_MU_S_SIZE = 32 x 8
    // SCATTERING_TEXTURE_HEIGHT: 128 = SCATTERING_TEXTURE_MU_SIZE
    // SCATTERING_TEXTURE_DEPTH: 32 = SCATTERING_TEXTURE_R_SIZE

    constexpr int32_t SCATTERING_TEXTURE_R_SIZE = 32;
    constexpr int32_t SCATTERING_TEXTURE_MU_SIZE = 128;
    constexpr int32_t SCATTERING_TEXTURE_MU_S_SIZE = 32;
    constexpr int32_t SCATTERING_TEXTURE_NU_SIZE = 8;

    constexpr int32_t SCATTERING_TEXTURE_WIDTH =
        SCATTERING_TEXTURE_NU_SIZE * SCATTERING_TEXTURE_MU_S_SIZE;
    constexpr int32_t SCATTERING_TEXTURE_HEIGHT = SCATTERING_TEXTURE_MU_SIZE;
    constexpr int32_t SCATTERING_TEXTURE_DEPTH = SCATTERING_TEXTURE_R_SIZE;

    // We store T(r,µ) and E(r,µ) in 64x256 and 16x64 textures.
    // For E(r,µ), 16x64 textures.
    // IRRADIANCE_TEXTURE_WIDTH: 64
    // IRRADIANCE_TEXTURE_HEIGHT: 16

    constexpr int32_t IRRADIANCE_TEXTURE_WIDTH = 64;
    constexpr int32_t IRRADIANCE_TEXTURE_HEIGHT = 16;

    // The conversion factor between watts and lumens.
    // Maximum luminous efficacy.
    // 最大視感効率. 683 [lumen/watt].
    constexpr float MAX_LUMINOUS_EFFICACY = 683.0F;

    // Values from "CIE (1931) 2-deg color matching functions", see
    // "http://web.archive.org/web/20081228084047/
    //    http://www.cvrl.org/database/data/cmfs/ciexyz31.txt".
    // CIE1931XYZ表色系2度視野の等色関数.
    constexpr auto NumOfElem = (LambdaMax - LambdaMin) / 5 + 1;
    struct CIE_2_DEG_COLOR_MATCHING_FUNCTIONS_ELEMENT {
        int32_t labmda;
        float x, y, z;
    };

    constexpr std::array<CIE_2_DEG_COLOR_MATCHING_FUNCTIONS_ELEMENT, NumOfElem> CIE_2_DEG_COLOR_MATCHING_FUNCTIONS = {
        {
            { 360, 0.000129900000F, 0.000003917000F, 0.000606100000F },
            { 365, 0.000232100000F, 0.000006965000F, 0.001086000000F },
            { 370, 0.000414900000F, 0.000012390000F, 0.001946000000F },
            { 375, 0.000741600000F, 0.000022020000F, 0.003486000000F },
            { 380, 0.001368000000F, 0.000039000000F, 0.006450001000F },
            { 385, 0.002236000000F, 0.000064000000F, 0.010549990000F },
            { 390, 0.004243000000F, 0.000120000000F, 0.020050010000F },
            { 395, 0.007650000000F, 0.000217000000F, 0.036210000000F },
            { 400, 0.014310000000F, 0.000396000000F, 0.067850010000F },
            { 405, 0.023190000000F, 0.000640000000F, 0.110200000000F },
            { 410, 0.043510000000F, 0.001210000000F, 0.207400000000F },
            { 415, 0.077630000000F, 0.002180000000F, 0.371300000000F },
            { 420, 0.134380000000F, 0.004000000000F, 0.645600000000F },
            { 425, 0.214770000000F, 0.007300000000F, 1.039050100000F },
            { 430, 0.283900000000F, 0.011600000000F, 1.385600000000F },
            { 435, 0.328500000000F, 0.016840000000F, 1.622960000000F },
            { 440, 0.348280000000F, 0.023000000000F, 1.747060000000F },
            { 445, 0.348060000000F, 0.029800000000F, 1.782600000000F },
            { 450, 0.336200000000F, 0.038000000000F, 1.772110000000F },
            { 455, 0.318700000000F, 0.048000000000F, 1.744100000000F },
            { 460, 0.290800000000F, 0.060000000000F, 1.669200000000F },
            { 465, 0.251100000000F, 0.073900000000F, 1.528100000000F },
            { 470, 0.195360000000F, 0.090980000000F, 1.287640000000F },
            { 475, 0.142100000000F, 0.112600000000F, 1.041900000000F },
            { 480, 0.095640000000F, 0.139020000000F, 0.812950100000F },
            { 485, 0.057950010000F, 0.169300000000F, 0.616200000000F },
            { 490, 0.032010000000F, 0.208020000000F, 0.465180000000F },
            { 495, 0.014700000000F, 0.258600000000F, 0.353300000000F },
            { 500, 0.004900000000F, 0.323000000000F, 0.272000000000F },
            { 505, 0.002400000000F, 0.407300000000F, 0.212300000000F },
            { 510, 0.009300000000F, 0.503000000000F, 0.158200000000F },
            { 515, 0.029100000000F, 0.608200000000F, 0.111700000000F },
            { 520, 0.063270000000F, 0.710000000000F, 0.078249990000F },
            { 525, 0.109600000000F, 0.793200000000F, 0.057250010000F },
            { 530, 0.165500000000F, 0.862000000000F, 0.042160000000F },
            { 535, 0.225749900000F, 0.914850100000F, 0.029840000000F },
            { 540, 0.290400000000F, 0.954000000000F, 0.020300000000F },
            { 545, 0.359700000000F, 0.980300000000F, 0.013400000000F },
            { 550, 0.433449900000F, 0.994950100000F, 0.008749999000F },
            { 555, 0.512050100000F, 1.000000000000F, 0.005749999000F },
            { 560, 0.594500000000F, 0.995000000000F, 0.003900000000F },
            { 565, 0.678400000000F, 0.978600000000F, 0.002749999000F },
            { 570, 0.762100000000F, 0.952000000000F, 0.002100000000F },
            { 575, 0.842500000000F, 0.915400000000F, 0.001800000000F },
            { 580, 0.916300000000F, 0.870000000000F, 0.001650001000F },
            { 585, 0.978600000000F, 0.816300000000F, 0.001400000000F },
            { 590, 1.026300000000F, 0.757000000000F, 0.001100000000F },
            { 595, 1.056700000000F, 0.694900000000F, 0.001000000000F },
            { 600, 1.062200000000F, 0.631000000000F, 0.000800000000F },
            { 605, 1.045600000000F, 0.566800000000F, 0.000600000000F },
            { 610, 1.002600000000F, 0.503000000000F, 0.000340000000F },
            { 615, 0.938400000000F, 0.441200000000F, 0.000240000000F },
            { 620, 0.854449900000F, 0.381000000000F, 0.000190000000F },
            { 625, 0.751400000000F, 0.321000000000F, 0.000100000000F },
            { 630, 0.642400000000F, 0.265000000000F, 0.000049999990F },
            { 635, 0.541900000000F, 0.217000000000F, 0.000030000000F },
            { 640, 0.447900000000F, 0.175000000000F, 0.000020000000F },
            { 645, 0.360800000000F, 0.138200000000F, 0.000010000000F },
            { 650, 0.283500000000F, 0.107000000000F, 0.000000000000F },
            { 655, 0.218700000000F, 0.081600000000F, 0.000000000000F },
            { 660, 0.164900000000F, 0.061000000000F, 0.000000000000F },
            { 665, 0.121200000000F, 0.044580000000F, 0.000000000000F },
            { 670, 0.087400000000F, 0.032000000000F, 0.000000000000F },
            { 675, 0.063600000000F, 0.023200000000F, 0.000000000000F },
            { 680, 0.046770000000F, 0.017000000000F, 0.000000000000F },
            { 685, 0.032900000000F, 0.011920000000F, 0.000000000000F },
            { 690, 0.022700000000F, 0.008210000000F, 0.000000000000F },
            { 695, 0.015840000000F, 0.005723000000F, 0.000000000000F },
            { 700, 0.011359160000F, 0.004102000000F, 0.000000000000F },
            { 705, 0.008110916000F, 0.002929000000F, 0.000000000000F },
            { 710, 0.005790346000F, 0.002091000000F, 0.000000000000F },
            { 715, 0.004109457000F, 0.001484000000F, 0.000000000000F },
            { 720, 0.002899327000F, 0.001047000000F, 0.000000000000F },
            { 725, 0.002049190000F, 0.000740000000F, 0.000000000000F },
            { 730, 0.001439971000F, 0.000520000000F, 0.000000000000F },
            { 735, 0.000999949300F, 0.000361100000F, 0.000000000000F },
            { 740, 0.000690078600F, 0.000249200000F, 0.000000000000F },
            { 745, 0.000476021300F, 0.000171900000F, 0.000000000000F },
            { 750, 0.000332301100F, 0.000120000000F, 0.000000000000F },
            { 755, 0.000234826100F, 0.000084800000F, 0.000000000000F },
            { 760, 0.000166150500F, 0.000060000000F, 0.000000000000F },
            { 765, 0.000117413000F, 0.000042400000F, 0.000000000000F },
            { 770, 0.000083075270F, 0.000030000000F, 0.000000000000F },
            { 775, 0.000058706520F, 0.000021200000F, 0.000000000000F },
            { 780, 0.000041509940F, 0.000014990000F, 0.000000000000F },
            { 785, 0.000029353260F, 0.000010600000F, 0.000000000000F },
            { 790, 0.000020673830F, 0.000007465700F, 0.000000000000F },
            { 795, 0.000014559770F, 0.000005257800F, 0.000000000000F },
            { 800, 0.000010253980F, 0.000003702900F, 0.000000000000F },
            { 805, 0.000007221456F, 0.000002607800F, 0.000000000000F },
            { 810, 0.000005085868F, 0.000001836600F, 0.000000000000F },
            { 815, 0.000003581652F, 0.000001293400F, 0.000000000000F },
            { 820, 0.000002522525F, 0.000000910930F, 0.000000000000F },
            { 825, 0.000001776509F, 0.000000641530F, 0.000000000000F },
            { 830, 0.000001251141F, 0.000000451810F, 0.000000000000F },
        }
    };

    // The conversion matrix from XYZ to linear sRGB color spaces.
    // Values from https://en.wikipedia.org/wiki/SRGB.
    constexpr std::array XYZ_TO_SRGB = {
      +3.2406F, -1.5372F, -0.4986F,
      -0.9689F, +1.8758F, +0.0415F,
      +0.0557F, -0.2040F, +1.0570F
    };

}
