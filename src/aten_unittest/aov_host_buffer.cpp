#include <gtest/gtest.h>

#include "aten.h"
#include "renderer/aov.h"

TEST(aten_test, aov_host_buffer) {
    aten::AOVHostBuffer<std::vector<aten::vec4>, 2> aov;
}
