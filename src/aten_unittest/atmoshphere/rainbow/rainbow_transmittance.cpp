#include "rainbow_test_fixture.h"

TEST_F(RainbowTest, GetRMuFromTransmittanceTextureUv)
{
    static const aten::vec2 TRANSMITTANCE_TEXTURE_SIZE{
        aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
        aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT,
    };

    int32_t x = 0;
    int32_t y = 0;

    aten::vec2 frag_coord{
        static_cast<float>(x) + 0.5F,
        static_cast<float>(y) + 0.5F,
    };

    float r;
    float mu;

    aten::rainbow::GetRMuFromTransmittanceTextureUv(
        atmosphere_, rain_volume_,
        frag_coord / TRANSMITTANCE_TEXTURE_SIZE,
        r, mu);

    auto uv = aten::rainbow::GetTransmittanceTextureUvFromRMu(
        atmosphere_, rain_volume_,
        r, mu);

    auto _x = uv.x * TRANSMITTANCE_TEXTURE_SIZE.x - 0.5F;
    auto _y = uv.y * TRANSMITTANCE_TEXTURE_SIZE.y - 0.5F;

    x = 128;
    y = 32;

    frag_coord = aten::vec2{
        static_cast<float>(x) + 0.5F,
        static_cast<float>(y) + 0.5F,
    };

    aten::rainbow::GetRMuFromTransmittanceTextureUv(
        atmosphere_, rain_volume_,
        frag_coord / TRANSMITTANCE_TEXTURE_SIZE,
        r, mu);

    uv = aten::rainbow::GetTransmittanceTextureUvFromRMu(
        atmosphere_, rain_volume_,
        r, mu);

    _x = uv.x * TRANSMITTANCE_TEXTURE_SIZE.x - 0.5F;
    _y = uv.y * TRANSMITTANCE_TEXTURE_SIZE.y - 0.5F;

    x = 0;
    y = 63;

    frag_coord = aten::vec2{
        static_cast<float>(x) + 0.5F,
        static_cast<float>(y) + 0.5F,
    };

    aten::rainbow::GetRMuFromTransmittanceTextureUv(
        atmosphere_, rain_volume_,
        frag_coord / TRANSMITTANCE_TEXTURE_SIZE,
        r, mu);

    uv = aten::rainbow::GetTransmittanceTextureUvFromRMu(
        atmosphere_, rain_volume_,
        r, mu);

    _x = uv.x * TRANSMITTANCE_TEXTURE_SIZE.x - 0.5F;
    _y = uv.y * TRANSMITTANCE_TEXTURE_SIZE.y - 0.5F;
}
