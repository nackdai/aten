#pragma once

#include "sky_test_fixture.h"
#include "image/texture.h"

class LazyTexture2D : public aten::texture {
protected:
    LazyTexture2D() = default;
    LazyTexture2D(int32_t width, int32_t height)
    {
        value_.resize(width * height, aten::vec4(-1.0F));

        width_ = width;
        height_ = height;
        m_size = width_ * height_;
    }

public:
    void Clear()
    {
        std::fill(value_.begin(), value_.end(), aten::vec4(-1.0F));
    }
};

class LazyTexture3D : public aten::texture3d {
protected:
    LazyTexture3D() = default;
    LazyTexture3D(int32_t width, int32_t height, int32_t depth)
    {
        value_.resize(width * height * depth, aten::vec4(-1.0F));

        width_ = width;
        height_ = height;
        depth_ = depth;
    }

public:
    void Clear()
    {
        std::fill(value_.begin(), value_.end(), aten::vec4(-1.0F));
    }
};

class LazyTransmittanceTexture : public LazyTexture2D {
public:
    LazyTransmittanceTexture() = default;
    explicit LazyTransmittanceTexture(
        const aten::sky::AtmosphereParameters& atmosphere)
        : LazyTexture2D(aten::sky::TRANSMITTANCE_TEXTURE_WIDTH, aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT),
        atmosphere_(atmosphere)
    {}

    static aten::vec3 ComputeTransmittanceToTopAtmosphereBoundaryTexture(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::vec2& screen_coord)
    {
        const aten::vec2 TRANSMITTANCE_TEXTURE_SIZE{
            aten::sky::TRANSMITTANCE_TEXTURE_WIDTH,
            aten::sky::TRANSMITTANCE_TEXTURE_HEIGHT };

        float r;
        float mu;
        aten::sky::GetRMuFromTransmittanceTextureUv(
            atmosphere,
            screen_coord / TRANSMITTANCE_TEXTURE_SIZE,
            r, mu);

        return aten::sky::transmittance::ComputeTransmittanceToTopAtmosphereBoundary(atmosphere, r, mu);
    }

    aten::vec4 AtByXY(int32_t x, int32_t y) const override
    {
        int32_t index = x + y * aten::sky::TRANSMITTANCE_TEXTURE_WIDTH;
        if (value_[index][0] < 0.0F) {
            const_cast<std::vector<aten::vec4>&>(value_)[index] = ComputeTransmittanceToTopAtmosphereBoundaryTexture(
                atmosphere_, aten::vec2(x + 0.5F, y + 0.5F));
        }
        return value_[index];
    }

private:
    const aten::sky::AtmosphereParameters& atmosphere_;
};

class LazySingleScatteringTexture : public LazyTexture3D {
public:
    LazySingleScatteringTexture(
        const aten::sky::AtmosphereParameters& atmosphere,
        const LazyTransmittanceTexture& transmittance_texture,
        bool is_rayleigh)
        : LazyTexture3D(aten::sky::SCATTERING_TEXTURE_WIDTH, aten::sky::SCATTERING_TEXTURE_HEIGHT, aten::sky::SCATTERING_TEXTURE_DEPTH),
        atmosphere_(atmosphere),
        transmittance_texture_(transmittance_texture),
        is_rayleigh_(is_rayleigh)
    {}

    static void ComputeSingleScatteringTexture(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::texture& transmittance_texture,
        const aten::vec3& frag_coord,
        aten::vec3& rayleigh,
        aten::vec3& mie)
    {
        float r;
        float mu;
        float mu_s;
        float nu;
        bool ray_r_mu_intersects_ground;

        aten::sky::GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, frag_coord,
            r, mu, mu_s, nu, ray_r_mu_intersects_ground);

        aten::sky::single_scattering::ComputeSingleScattering(atmosphere, transmittance_texture,
            r, mu, mu_s, nu, ray_r_mu_intersects_ground, rayleigh, mie);
    }


    aten::vec4 AtByXYZ(int32_t x, int32_t y, int32_t z) const override
    {
        int32_t index = x + aten::sky::SCATTERING_TEXTURE_WIDTH * (y + aten::sky::SCATTERING_TEXTURE_HEIGHT * z);
        if (value_[index][0] < 0.0 * watt_per_square_meter_per_nm) {
            aten::vec3 rayleigh{ 0.0F };
            aten::vec3 mie{ 0.0F };
            ComputeSingleScatteringTexture(
                atmosphere_, transmittance_texture_,
                aten::vec3(x + 0.5F, y + 0.5F, z + 0.5F),
                rayleigh, mie);
            const_cast<std::vector<aten::vec4>&>(value_)[index] = is_rayleigh_ ? rayleigh : mie;
        }
        return value_[index];
    }

private:
    const aten::sky::AtmosphereParameters& atmosphere_;
    const LazyTransmittanceTexture& transmittance_texture_;
    bool is_rayleigh_;
};

class LazyScatteringDensityTexture : public LazyTexture3D {
public:
    LazyScatteringDensityTexture(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::texture& transmittance_texture,
        const aten::texture3d& single_rayleigh_scattering_texture,
        const aten::texture3d& single_mie_scattering_texture,
        const aten::texture3d& multiple_scattering_texture,
        const aten::texture& irradiance_texture,
        const int order)
        : LazyTexture3D(aten::sky::SCATTERING_TEXTURE_WIDTH, aten::sky::SCATTERING_TEXTURE_HEIGHT, aten::sky::SCATTERING_TEXTURE_DEPTH),
        atmosphere_(atmosphere),
        transmittance_texture_(transmittance_texture),
        single_rayleigh_scattering_texture_(single_rayleigh_scattering_texture),
        single_mie_scattering_texture_(single_mie_scattering_texture),
        multiple_scattering_texture_(multiple_scattering_texture),
        irradiance_texture_(irradiance_texture),
        order_(order)
    {}

    static aten::vec3 ComputeScatteringDensityTexture(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::texture& transmittance_texture,
        const aten::texture3d& single_rayleigh_scattering_texture,
        const aten::texture3d& single_mie_scattering_texture,
        const aten::texture3d& multiple_scattering_texture,
        const aten::texture& irradiance_texture,
        const aten::vec3& frag_coord,
        int scattering_order)
    {
        float r;
        float mu;
        float mu_s;
        float nu;
        bool ray_r_mu_intersects_ground;
        aten::sky::GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, frag_coord,
            r, mu, mu_s, nu, ray_r_mu_intersects_ground);

        return aten::sky::ComputeScatteringDensity(atmosphere, transmittance_texture,
            single_rayleigh_scattering_texture, single_mie_scattering_texture,
            multiple_scattering_texture, irradiance_texture, r, mu, mu_s, nu,
            scattering_order);
    }


    aten::vec4 AtByXYZ(int32_t x, int32_t y, int32_t z) const override
    {
        int32_t index = x + aten::sky::SCATTERING_TEXTURE_WIDTH * (y + aten::sky::SCATTERING_TEXTURE_HEIGHT * z);
        if (value_[index][0] < 0.0F) {
            const_cast<std::vector<aten::vec4>&>(value_)[index] = ComputeScatteringDensityTexture(
                atmosphere_, transmittance_texture_,
                single_rayleigh_scattering_texture_, single_mie_scattering_texture_,
                multiple_scattering_texture_, irradiance_texture_,
                aten::vec3(x + 0.5F, y + 0.5F, z + 0.5F), order_);
        }
        return value_[index];
    }

private:
    const aten::sky::AtmosphereParameters& atmosphere_;
    const aten::texture& transmittance_texture_;
    const aten::texture3d& single_rayleigh_scattering_texture_;
    const aten::texture3d& single_mie_scattering_texture_;
    const aten::texture3d& multiple_scattering_texture_;
    const aten::texture& irradiance_texture_;
    const int order_;
};

class LazyMultipleScatteringTexture : public LazyTexture3D {
public:
    LazyMultipleScatteringTexture(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::texture& transmittance_texture,
        const aten::texture3d& scattering_density_texture)
        : LazyTexture3D(aten::sky::SCATTERING_TEXTURE_WIDTH, aten::sky::SCATTERING_TEXTURE_HEIGHT, aten::sky::SCATTERING_TEXTURE_DEPTH),
        atmosphere_(atmosphere),
        transmittance_texture_(transmittance_texture),
        scattering_density_texture_(scattering_density_texture) {
    }

    static aten::vec3 ComputeMultipleScatteringTexture(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::texture& transmittance_texture,
        const aten::texture3d& scattering_density_texture,
        const aten::vec3& frag_coord,
        float& nu)
    {
        float r;
        float mu;
        float mu_s;
        bool ray_r_mu_intersects_ground;
        aten::sky::GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, frag_coord,
            r, mu, mu_s, nu, ray_r_mu_intersects_ground);

        return aten::sky::ComputeMultipleScattering(atmosphere, transmittance_texture,
            scattering_density_texture, r, mu, mu_s, nu,
            ray_r_mu_intersects_ground);
    }


    aten::vec4 AtByXYZ(int32_t x, int32_t y, int32_t z) const override
    {
        int32_t index = x + aten::sky::SCATTERING_TEXTURE_WIDTH * (y + aten::sky::SCATTERING_TEXTURE_HEIGHT * z);
        if (value_[index][0] < 0.0F) {
            float ignored;
            const_cast<std::vector<aten::vec4>&>(value_)[index] = ComputeMultipleScatteringTexture(
                atmosphere_,
                transmittance_texture_, scattering_density_texture_,
                aten::vec3(x + 0.5F, y + 0.5F, z + 0.5F), ignored);
        }
        return value_[index];
    }

private:
    const aten::sky::AtmosphereParameters& atmosphere_;
    const aten::texture& transmittance_texture_;
    const aten::texture3d& scattering_density_texture_;
};

class LazyIndirectIrradianceTexture : public LazyTexture2D {
public:
    LazyIndirectIrradianceTexture(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::texture3d& single_rayleigh_scattering_texture,
        const aten::texture3d& single_mie_scattering_texture,
        const aten::texture3d& multiple_scattering_texture,
        int scattering_order)
        : LazyTexture2D(aten::sky::IRRADIANCE_TEXTURE_WIDTH, aten::sky::IRRADIANCE_TEXTURE_HEIGHT),
        atmosphere_(atmosphere),
        single_rayleigh_scattering_texture_(single_rayleigh_scattering_texture),
        single_mie_scattering_texture_(single_mie_scattering_texture),
        multiple_scattering_texture_(multiple_scattering_texture),
        scattering_order_(scattering_order) {
    }

    static aten::vec3 ComputeIndirectIrradianceTexture(
        const aten::sky::AtmosphereParameters& atmosphere,
        const aten::texture3d& single_rayleigh_scattering_texture,
        const aten::texture3d& single_mie_scattering_texture,
        const aten::texture3d& multiple_scattering_texture,
        const aten::vec2& frag_coord,
        int scattering_order)
    {
        const aten::vec2 IRRADIANCE_TEXTURE_SIZE{
            aten::sky::IRRADIANCE_TEXTURE_WIDTH,
            aten::sky::IRRADIANCE_TEXTURE_HEIGHT };

        float r;
        float mu_s;
        aten::sky::GetRMuSFromIrradianceTextureUv(
            atmosphere, frag_coord / IRRADIANCE_TEXTURE_SIZE, r, mu_s);

        return aten::sky::ComputeIndirectIrradiance(atmosphere,
            single_rayleigh_scattering_texture, single_mie_scattering_texture,
            multiple_scattering_texture, r, mu_s, scattering_order);
    }

    aten::vec4 AtByXY(int32_t x, int32_t y) const override
    {
        //int32_t index = x + y * aten::sky::IRRADIANCE_TEXTURE_WIDTH;
        int32_t index = x + y * aten::sky::IRRADIANCE_TEXTURE_HEIGHT;
        if (value_[index][0] < 0.0) {
            const_cast<std::vector<aten::vec4>&>(value_)[index] = ComputeIndirectIrradianceTexture(
                atmosphere_,
                single_rayleigh_scattering_texture_,
                single_mie_scattering_texture_,
                multiple_scattering_texture_,
                aten::vec2(x + 0.5F, y + 0.5F),
                scattering_order_);
        }
        return value_[index];
    }

private:
    const aten::sky::AtmosphereParameters& atmosphere_;
    const aten::texture3d& single_rayleigh_scattering_texture_;
    const aten::texture3d& single_mie_scattering_texture_;
    const aten::texture3d& multiple_scattering_texture_;
    int scattering_order_;
};
