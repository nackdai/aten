#include "mirrormap.h"

#include "atenscene.h"

std::tuple<float, float, CubemapFace> MirrorMap::GetUVFromDir(const aten::vec3& dir) const
{
    // We assume the followings:
    //  * Mirror ball
    //    * We can see the reflected scene on the surface of the mirror ball.
    //    * So, we can't see the scene at the behind of the mirror ball on the surface of the mirror ball.
    //
    //         +y /+x                        +v
    //         | /                            |
    //         |/                            /
    //  scene  +-------->z   @ mirror ball  /+u
    //
    //  * The direction of +x is -u direction. So, we need to flip u.
    //
    //  * The vector which is reflected by the normal on the mirror ball is the direction to the scene.
    //  * If we define the normal on the mirror ball as N = (N.x, N.y, N.z), N.x and N.y are (u, v) coordinate values.
    //  * reflection = dir

    // E(eye) = (0, 0, 1) (: Camera forward is along with the z-axis)
    // R(reflection) = 2 * dot(N, E) * N - E
    // dot(N, E) = dot((N.x, N.y, N.z), (0, 0, 1) = N.z

    // R = 2 * dot(N, eye) * N - E = 2 * N.z * (N.x, N.y, N.z) - (0, 0, 1)
    //  -> R.z = 2 * N.z * N.z - 1 = 2 * pow(N.z, 2) - 1
    //       => N.z = sqrt((R.z + 1) / 2)
    //  -> R.x = 2 * N.z * N.x - 0
    //       => N.x = R.x / (2 * N.z) = u
    //  -> R.y = 2 * N.z * N.y - 0
    //       => N.y = R.y / (2 * N.z) = v
    //
    // 2 * N.z = 2 * sqrt((R.z + 1) / 2) = sqrt(4 * (R.z + 1) / 2) = sqrt(2 * (R.z + 1))

    const auto& R = dir;
    const auto div = 1.0F / aten::sqrt(2.0F * (R.z + 1.0F));

    // The direction of +x is -u direction. So, we need to flip u.
    auto u = -R.x * div;
    auto v = R.y * div;

    u = aten::clamp(u, -1.0F, 1.0F);
    v = aten::clamp(v, -1.0F, 1.0F);

    // Normalize [-1, 1] -> [0, 1]
    u = (u + 1.0F) * 0.5F;
    v = (v + 1.0F) * 0.5F;

    return std::make_tuple(u, v, CubemapFace::Num);
}

aten::vec3 MirrorMap::GetDirFromXY(
    int32_t x, int32_t y,
    CubemapFace face/*= CubemapFace::Num*/) const
{
    // If we define the normal on the mirror ball as N = (N.x, N.y, N.z), N.x and N.y are (u, v) coordinate values.
    // It means we can map (u, v) as N.x and N.y.
    // And, we can compute N.z = sqrt(1 - N.x * N.x - N.y * N.y) = sqrt(1 - u * u - v * v)

    // We need to compute the reflection vector as the direction.
    //  E(eye) = (0, 0, 1) (: Camera forward is along with the z-axis)
    //  R(reflection) = 2 * dot(N, E) * N - E
    //  dot(N, E) = dot((N.x, N.y, N.z), (0, 0, 1) = N.z

    std::ignore = face;

    const auto width = static_cast<float>(tex_->width() - 1);
    const auto height = static_cast<float>(tex_->height() - 1);

    // NOTE:
    //         +y /+x                        +v
    //         | /                            |
    //         |/                            /
    //  scene  +-------->z   @ mirror ball  /+u
    //
    // The direction of +x is -u direction. So, we need to flip u.

    auto u = x / width;
    u = 1 - u;

    // [0, 1] -> [-1, 1].
    u = 2.0F * u - 1.0F;
    const auto v = 2.0F * (y / height) - 1.0F;

    aten::vec3 N(
        u,
        v,
        aten::sqrt(1.0F - u * u - v * v)
    );
    N = normalize(N);

    const aten::vec3 E(0, 0, 1);

    aten::vec3 dir = 2 * dot(N, E) * N - E;

    dir = normalize(dir);

    return dir;
}

bool MirrorMap::IsValidPos(int32_t x, int32_t y) const
{
    // We assume the width is equal with height;
    const auto width = this->width();
    const auto height = this->height();

    if (width != height) {
        AT_ASSERT(false);
        return false;
    }

    float u = x / static_cast<float>(width - 1);
    float v = y / static_cast<float>(height - 1);

    // [0, 1] -> [-1, 1]
    u = 2.0F * u - 1.0F;
    v = 2.0F * v - 1.0F;

    return ((u * u + v * v) <= 1.0F);
}
