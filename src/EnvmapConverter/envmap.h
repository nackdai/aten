#pragma once

#include <memory>
#include <string>
#include <tuple>

#include "aten.h"

enum class EnvMapType : int {
    Equirect,
    Angular,
    Mirror,
    CubeMap,
    Invalid,
};

enum class CubemapFace : int {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
    Num,
};

class EnvMap {
protected:
    EnvMap(EnvMapType type) : type_(type) {}

    EnvMap(const EnvMap&) = delete;
    EnvMap(EnvMap&&) = delete;
    EnvMap& operator=(const EnvMap&) = delete;
    EnvMap& operator=(EnvMap&&) = delete;

public:
    // To pass the polymorphic raw pointer to shared_ptr.
    EnvMap() = default;
    virtual ~EnvMap() = default;

    static std::shared_ptr<EnvMap> LoadEnvmap(
        aten::context& ctxt,
        EnvMapType type,
        std::string_view filename,
        std::string_view filename_neg_x = "",
        std::string_view filename_pos_y = "",
        std::string_view filename_neg_y = "",
        std::string_view filename_pos_z = "",
        std::string_view filename_neg_z = "");

    static std::shared_ptr<EnvMap> CreateEmptyEnvmap(
        EnvMapType type,
        std::int32_t width, std::int32_t height);

    static void Convert(
        const std::shared_ptr<EnvMap>& src,
        const std::shared_ptr<EnvMap>& dst);

    virtual std::tuple<float, float, CubemapFace> GetUVFromDir(const aten::vec3& dir) const = 0;

    virtual aten::vec3 GetDirFromXY(
        int32_t x, int32_t y,
        CubemapFace face = CubemapFace::Num) const = 0;

    virtual aten::vec4 At(
        float u, float v,
        CubemapFace face = CubemapFace::Num) const = 0;

    virtual aten::vec4 GetColorFromDir(const aten::vec3& dir) const
    {
        const auto [u, v, face] = GetUVFromDir(dir);
        return At(u, v, face);
    }

    virtual void Put(
        const aten::vec4& color,
        int32_t x, int32_t y,
        CubemapFace face = CubemapFace::Num) = 0;

    virtual bool SaveAsPng(std::string_view filename) const = 0;

    virtual int32_t width() const = 0;

    virtual int32_t height() const = 0;

    virtual bool IsValidPos(int32_t x, int32_t y) const
    {
        return true;
    }

    EnvMapType type() const
    {
        return type_;
    }

protected:
    EnvMapType type_;
};

class SingleEnvMap : public EnvMap {
protected:
    SingleEnvMap(EnvMapType type) : EnvMap(type) {}

    SingleEnvMap(const SingleEnvMap&) = delete;
    SingleEnvMap(SingleEnvMap&&) = delete;
    SingleEnvMap& operator=(const SingleEnvMap&) = delete;
    SingleEnvMap& operator=(SingleEnvMap&&) = delete;

public:
    template<typename TEnvMap>
    static TEnvMap* Load(aten::context& ctxt, std::string_view filename);

    template<typename TEnvMap>
    static TEnvMap* Create(std::int32_t width, std::int32_t height);

    aten::vec4 At(
        float u, float v,
        CubemapFace face = CubemapFace::Num) const override;

    void Put(
        const aten::vec4& color,
        std::int32_t x, std::int32_t y,
        CubemapFace face = CubemapFace::Num) override;

    bool SaveAsPng(std::string_view filename) const override;

    int32_t width() const override
    {
        AT_ASSERT(tex_);
        if (!tex_) {
            return 0;
        }
        return tex_->width();
    }

    int32_t height() const override
    {
        AT_ASSERT(tex_);
        if (!tex_) {
            return 0;
        }
        return tex_->height();
    }

protected:
    std::shared_ptr<aten::texture> tex_;
};
