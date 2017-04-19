#include "kernel/raytracing.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten.h"

// 直行ベクトルを計算.
__host__ __device__ aten::vec3 getOrthoVector(const aten::vec3& n)
{
	aten::vec3 p;

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

	aten::vec3 p;

	aten::vec3 normal;

	int mtrlid;

	float area{ 1.0f };
};

struct Sphere {
	float radius;
	aten::vec3 center;
	int mtrlid;

	Sphere() {}
	Sphere(const aten::vec3& c, float r, int id)
	{
		radius = r;
		center = c;
		mtrlid = id;
	}
};

struct Context {
	int geomnum;
	Sphere* spheres;

	aten::MaterialParameter* mtrls;
};

__host__ __device__ bool intersectSphere(
	const Sphere* sphere,
	const aten::ray* r, hitrecord* rec)
{
	const aten::vec3 p_o = sphere->center - r->org;
	const float b = dot(p_o, r->dir);

	// 判別式.
	const float D4 = b * b - dot(p_o, p_o) + sphere->radius * sphere->radius;

	if (D4 < 0.0f) {
		return false;
	}

	const float sqrt_D4 = sqrtf(D4);
	const float t1 = b - sqrt_D4;
	const float t2 = b + sqrt_D4;

	if (t1 > AT_MATH_EPSILON) {
		rec->t = t1;
	}
	else if (t2 > AT_MATH_EPSILON) {
		rec->t = t2;
	}
	else {
		return false;
	}

	rec->p = r->org + rec->t * r->dir;
	rec->normal = (rec->p - sphere->center) / sphere->radius; // 正規化して法線を得る

	rec->mtrlid = sphere->mtrlid;

	rec->area = 4 * AT_MATH_PI * sphere->radius * sphere->radius;

	return true;
}

struct CameraSampleResult {
	aten::ray r;
	aten::vec3 posOnLens;
	aten::vec3 nmlOnLens;

	__host__ __device__ CameraSampleResult() {}
};

struct Camera {
	aten::vec3 origin;

	float aspect;
	aten::vec3 center;

	aten::vec3 u;
	aten::vec3 v;

	aten::vec3 dir;
	aten::vec3 right;
	aten::vec3 up;

	float dist;
	int width;
	int height;
};

__host__ void initCamera(
	Camera& camera,
	const aten::vec3& origin,
	const aten::vec3& lookat,
	const aten::vec3& up,
	float vfov,	// vertical fov.
	uint32_t width, uint32_t height)
{
	float theta = Deg2Rad(vfov);

	camera.aspect = width / (float)height;

	float half_height = tanf(theta / 2);
	float half_width = camera.aspect * half_height;

	camera.origin = origin;

	// カメラ座標ベクトル.
	camera.dir = normalize(lookat - origin);
	camera.right = normalize(cross(camera.dir, up));
	camera.up = cross(camera.right, camera.dir);

	camera.center = origin + camera.dir;

	// スクリーンのUVベクトル.
	camera.u = half_width * camera.right;
	camera.v = half_height * camera.up;

	camera.dist = height / (2.0f * tanf(theta / 2));

	camera.width = width;
	camera.height = height;
}

__host__ __device__ void sampleCamera(
	CameraSampleResult* sample,
	Camera* camera,
	float s, float t)
{
	// [0, 1] -> [-1, 1]
	s = 2 * s - 1;
	t = 2 * t - 1;

	auto screenPos = s * camera->u + t * camera->v;

	screenPos = screenPos + camera->center;

	auto dirToScr = screenPos - camera->origin;

	sample->posOnLens = screenPos;
	sample->nmlOnLens = camera->dir;
	sample->r = aten::ray(camera->origin, dirToScr);
}

__host__ __device__ bool intersect(
	const aten::ray* r, hitrecord* rec,
	const Context* ctx)
{
	bool isHit = false;

	hitrecord tmp;

	for (int i = 0; i < ctx->geomnum; i++) {
		if (intersectSphere(&ctx->spheres[i], r, &tmp)) {
			if (tmp.t < rec->t) {
				*rec = tmp;

				isHit = true;
			}
		}
	}

	return isHit;
}

#if 1
__global__ void raytracing(
	float4* p,
	int width, int height,
	Camera* camera,
	Sphere* spheres, int num,
	aten::MaterialParameter* mtrls)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;
#else
__host__ void raytracing(
	int ix, int iy,
	float4* p,
	int width, int height,
	Camera* camera,
	Sphere* spheres, int num)
{
#endif

	if (ix >= width && iy >= height) {
		return;
	}

	Context ctx;
	{
		ctx.geomnum = num;
		ctx.spheres = spheres;
	}

	const auto idx = iy * camera->width + ix;

	float s = ix / (float)camera->width;
	float t = iy / (float)camera->height;

	CameraSampleResult camsample;
	sampleCamera(&camsample, camera, s, t);

	hitrecord rec;

	if (intersect(&camsample.r, &rec, &ctx)) {
		const aten::MaterialParameter& m = mtrls[rec.mtrlid];
		p[idx] = make_float4(m.baseColor.x, m.baseColor.y, m.baseColor.z, 1);
	}
	else {
		p[idx] = make_float4(0, 0, 0, 1);
	}
}

static Sphere g_spheres[] = {
	Sphere(aten::vec3(0, 0, -10), 1.0, 0),
	Sphere(aten::vec3(3, 0, -10), 1.0, 1),
};

void renderRayTracing(
	aten::vec4* image,
	int width, int height,
	const std::vector<aten::material*>& mtrls)
{
	Camera camera;
	initCamera(
		camera,
		aten::vec3(0, 0, 0),
		aten::vec3(0, 0, -1),
		aten::vec3(0, 1, 0),
		30,
		width, height);

#if 1
	aten::CudaMemory dst(sizeof(float4) * width * height);

	aten::TypedCudaMemory<Camera> cam(&camera, 1);
	
	aten::TypedCudaMemory<Sphere> spheres(AT_COUNTOF(g_spheres));
	spheres.writeByNum(g_spheres, AT_COUNTOF(g_spheres));

	std::vector<aten::MaterialParameter> mtrlparams;
	for (auto m : mtrls) {
		mtrlparams.push_back(m->param());
	}

	aten::TypedCudaMemory<aten::MaterialParameter> materials(mtrlparams.size());
	materials.writeByNum(&mtrlparams[0], mtrlparams.size());

	dim3 block(32, 32);
	dim3 grid(
		(width + block.x - 1) / block.x,
		(height + block.y - 1) / block.y);

	raytracing << <grid, block >> > (
		(float4*)dst.ptr(), 
		width, height, 
		cam.ptr(),
		spheres.ptr(), AT_COUNTOF(g_spheres),
		materials.ptr());

	checkCudaErrors(cudaDeviceSynchronize());

	dst.read(image, sizeof(aten::vec4) * width * height);
#else
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			raytracing(
				x, y,
				(float4*)image,
				width, height,
				&camera,
				g_spheres, AT_COUNTOF(g_spheres));
		}
	}
#endif
}
