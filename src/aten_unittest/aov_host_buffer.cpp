#include <gtest/gtest.h>

#include "aten.h"
#include "renderer/aov.h"

TEST(aten_test, CompareAOVBufferTypeTest) {
    aten::AOVBufferType type;

    ASSERT_TRUE(type == aten::AOVBufferType::NormalDepth);
    ASSERT_FALSE(type != aten::AOVBufferType::NormalDepth);

    ASSERT_TRUE(type != aten::AOVBufferType::AlbedoMeshId);
    ASSERT_FALSE(type == aten::AOVBufferType::AlbedoMeshId);
}

TEST(aten_test, CompareAOVTypeTest) {
    aten::AOVType type;

    ASSERT_TRUE(type == aten::AOVType::Normal);
    ASSERT_FALSE(type != aten::AOVType::Normal);

    ASSERT_TRUE(type != aten::AOVType::Depth);
    ASSERT_FALSE(type == aten::AOVType::Depth);
}

TEST(aten_test, InheritAOVBufferTypeTest) {
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

TEST(aten_test, AOVHostBufferTest) {
    aten::AOVHostBuffer<std::vector<aten::vec4>, 2> aov;

    ASSERT_EQ(decltype(aov)::NumAOV, 2);

    int32_t count = 0;
    aov.traverse([&count](decltype(aov)::BufferType& buffer) {
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
