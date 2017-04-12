#include "kernel/raytracing.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda/helper_math.h"

#include "aten.h"

struct Ray {
	float3 org;
	float3 dir;

	Ray() {}
	Ray(float3 _org, float3 _dir)
	{
		org = _org;
		dir = _dir;
	}
};

// 直行ベクトルを計算.
__device__ float3 getOrthoVector(const float3& n)
{
	float3 p;

	// NOTE
	// dotを計算したときにゼロになるようなベクトル.
	// k は normalize 計算用.

	if (abs(n.z) > 0.0f) {
		float k = sqrtf(n.y * n.y + n.z * n.z);
		p.x = 0;
		p.y = -n.z / k;
		p.z = n.y / k;
	}
	else {
		float k = sqrtf(n.x * n.x + n.y * n.y);
		p.x = n.y / k;
		p.y = -n.x / k;
		p.z = 0;
	}

	return std::move(p);
}

struct hitrecord {
	float t{ AT_MATH_INF };

	float3 p;

	float3 normal;

	float area{ 1.0f };
};

class Sphere {
public:
	float radius;
	float3 center;
	float3 color;

	Sphere(float r, const float3& c, const float3& clr)
	{
		radius = r;
		center = c;
		color = clr;
	}

	__device__ bool intersect(const Ray& r, hitrecord& rec)
	{
		const float3 p_o = center - r.org;
		const float b = dot(p_o, r.dir);

		// 判別式.
		const float D4 = b * b - dot(p_o, p_o) + radius * radius;

		if (D4 < 0.0f) {
			return false;
		}

		const float sqrt_D4 = sqrtf(D4);
		const float t1 = b - sqrt_D4;
		const float t2 = b + sqrt_D4;

		if (t1 > AT_MATH_EPSILON) {
			rec.t = t1;
		}
		else if (t2 > AT_MATH_EPSILON) {
			rec.t = t2;
		}
		else {
			return false;
		}

		rec.p = r.org + rec.t * r.dir;
		rec.normal = (rec.p - center) / radius; // 正規化して法線を得る

		rec.area = 4 * AT_MATH_PI * radius * radius;

		return true;
	}
};

struct CameraSampleResult {
	Ray r;
	float3 posOnLens;
	float3 nmlOnLens;

	CameraSampleResult() {}
};

class Camera {
public:
	Camera() {}
	~Camera() {}

	__host__ __device__ void init(
		const float3& origin,
		const float3& lookat,
		const float3& up,
		float vfov,	// vertical fov.
		uint32_t width, uint32_t height)
	{
		float theta = Deg2Rad(vfov);

		m_aspect = width / (float)height;

		float half_height = tanf(theta / 2);
		float half_width = m_aspect * half_height;

		m_origin = origin;

		// カメラ座標ベクトル.
		m_dir = normalize(lookat - origin);
		m_right = normalize(cross(m_dir, up));
		m_up = cross(m_right, m_dir);

		m_center = origin + m_dir;

		// スクリーンのUVベクトル.
		m_u = half_width * m_right;
		m_v = half_height * m_up;

		m_dist = height / (2.0f * tanf(theta / 2));

		m_width = width;
		m_height = height;
	}

	__host__ __device__ CameraSampleResult sample(float s, float t)
	{
		CameraSampleResult result;

		// [0, 1] -> [-1, 1]
		s = 2 * s - 1;
		t = 2 * t - 1;

		auto screenPos = s * m_u + t * m_v;
		screenPos = screenPos + m_center;

		auto dirToScr = screenPos - m_origin;

		result.posOnLens = screenPos;
		result.nmlOnLens = m_dir;
		result.r = Ray(m_origin, dirToScr);

		return std::move(result);
	}

private:
	float3 m_origin;

	float m_aspect;
	float3 m_center;

	float3 m_u;
	float3 m_v;

	float3 m_dir;
	float3 m_right;
	float3 m_up;

	float m_dist;
	int m_width;
	int m_height;
};

__device__ bool intersect(
	const Ray& r, hitrecord& rec,
	Sphere* spheres, const int num)
{
	bool isHit = false;

	hitrecord tmp;

	for (int i = 0; i < num; i++) {
		if (spheres[i].intersect(r, tmp)) {
			if (tmp.t < rec.t) {
				rec = tmp;

				isHit = true;
			}
		}
	}

	return isHit;
}

__constant__ Camera camera;

__global__ void raytracing(
	int width, int height,
	Sphere* spheres, const int num)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	float s = ix / (float)width;
	float t = iy / (float)height;

	const auto camsample = camera.sample(s, t);

	hitrecord rec;

	if (intersect(camsample.r, rec, spheres, num)) {

	}
}

void sumArrayOnGPU(float* A, float* B, float* C, const int N)
{
}
