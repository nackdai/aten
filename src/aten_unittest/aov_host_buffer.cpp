#include <gtest/gtest.h>

#include "aten.h"
#include "renderer/aov.h"

TEST(aov_test, CompareAOVBufferTypeTest)
{
    aten::AOVBufferType type;

    ASSERT_TRUE(type == aten::AOVBufferType::NormalDepth);
    ASSERT_FALSE(type != aten::AOVBufferType::NormalDepth);

    ASSERT_TRUE(type != aten::AOVBufferType::AlbedoMeshId);
    ASSERT_FALSE(type == aten::AOVBufferType::AlbedoMeshId);
}

TEST(aov_test, CompareAOVTypeTest)
{
    aten::AOVType type;

    ASSERT_TRUE(type == aten::AOVType::Normal);
    ASSERT_FALSE(type != aten::AOVType::Normal);

    ASSERT_TRUE(type != aten::AOVType::Depth);
    ASSERT_FALSE(type == aten::AOVType::Depth);
}

TEST(aov_test, InheritAOVBufferTypeTest)
{
    struct Inheritance : public aten::AOVBufferType {
        enum Type {
            A = aten::AOVBufferType::BeginOfInheritType,
            B,
            C,
        };

        Inheritance() : aten::AOVBufferType(A) {}
        ~Inheritance() = default;
        Inheritance(int32_t type) : aten::AOVBufferType(static_cast<Type>(type)) {}
    };

    Inheritance type;

    ASSERT_TRUE(type == Inheritance::A);
    ASSERT_FALSE(type != Inheritance::A);
    ASSERT_TRUE(type != aten::AOVBufferType::NormalDepth);

    Inheritance type_base{ aten::AOVBufferType::NormalDepth };
    ASSERT_TRUE(type != aten::AOVBufferType::NormalDepth);
}

TEST(aov_test, AOVHostBufferTest)
{
    aten::AOVHostBuffer<std::vector<aten::vec4>, 2> aov;

    ASSERT_EQ(decltype(aov)::NumAOV, 2);

    int32_t count = 0;
    aov.traverse([&count](decltype(aov)::buffer_type& buffer) {
        buffer.push_back(aten::vec4(static_cast<float>(count)));
        count++;
    });

    ASSERT_EQ(aov.normal_depth().size(), 1);
    ASSERT_EQ(aov.normal_depth()[0].x, static_cast<real>(0));
    ASSERT_EQ(aov.get<aten::AOVBufferType::NormalDepth>()[0].x, static_cast<real>(0));

    ASSERT_EQ(aov.albedo_meshid().size(), 1);
    ASSERT_EQ(aov.albedo_meshid()[0].x, static_cast<real>(1));
    ASSERT_EQ(aov.get<aten::AOVBufferType::AlbedoMeshId>()[0].x, static_cast<real>(1));
}

TEST(aov_test, FillBasicAOVsTest)
{
    aten::vec4 aovNormalDepth;

    aten::vec3 normal(real(1), real(2), real(3));

    aten::hitrecord rec;
    rec.p = aten::vec4(real(1));

    aten::mat4 mtx_W2C;

    aten::vec4 aovAlbedoMeshId;

    aten::vec4 albedo(real(4), real(5), real(6), real(7));

    aten::Intersection isect;
    isect.meshid = 2;


    aten::FillBasicAOVs(
        aovNormalDepth,
        normal,
        rec,
        mtx_W2C,
        aovAlbedoMeshId,
        albedo,
        isect);

    ASSERT_EQ(aovNormalDepth.x, normal.x);
    ASSERT_EQ(aovNormalDepth.y, normal.y);
    ASSERT_EQ(aovNormalDepth.z, normal.z);
    ASSERT_EQ(aovNormalDepth.w, rec.p.z);

    ASSERT_EQ(aovAlbedoMeshId.x, albedo.x);
    ASSERT_EQ(aovAlbedoMeshId.y, albedo.y);
    ASSERT_EQ(aovAlbedoMeshId.z, albedo.z);
    ASSERT_EQ(aovAlbedoMeshId.w, isect.meshid);
}

TEST(aov_test, FillBasicAOVsIfHitMissTest)
{
    aten::vec4 aovNormalDepth;
    aten::vec4 aovAlbedoMeshId;

    aten::vec4 bg(real(4), real(5), real(6), real(7));

    aten::FillBasicAOVsIfHitMiss(
        aovNormalDepth,
        aovAlbedoMeshId,
        bg);

    ASSERT_EQ(aovNormalDepth.x, real(0));
    ASSERT_EQ(aovNormalDepth.y, real(0));
    ASSERT_EQ(aovNormalDepth.z, real(0));
    ASSERT_EQ(aovNormalDepth.w, real(-1));

    ASSERT_EQ(aovAlbedoMeshId.x, bg.x);
    ASSERT_EQ(aovAlbedoMeshId.y, bg.y);
    ASSERT_EQ(aovAlbedoMeshId.z, bg.z);
    ASSERT_EQ(aovAlbedoMeshId.w, real(-1));
}

TEST(aov_test, FillBaryCentricAOVTest)
{
    aten::vec4 aovBuffer;

    aten::Intersection isect;
    isect.a = real(0.1);
    isect.b = real(0.2);

    FillBaryCentricAOV(aovBuffer, isect);

    ASSERT_FLOAT_EQ(aovBuffer.x, isect.a);
    ASSERT_FLOAT_EQ(aovBuffer.y, isect.b);
    ASSERT_FLOAT_EQ(aovBuffer.z, real(1) - isect.a - isect.b);
}

TEST(aov_test, FillBaryCentricAOVIfHitMissTest)
{
    aten::vec4 aovBuffer;

    FillBaryCentricAOVIfHitMiss(aovBuffer);

    ASSERT_EQ(aovBuffer.x, real(0));
    ASSERT_EQ(aovBuffer.y, real(0));
    ASSERT_EQ(aovBuffer.z, real(0));
}
