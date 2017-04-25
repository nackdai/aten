#include "kernel/raytracing.h"
#include "kernel/light.cuh"
#include "kernel/material.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

struct Context {
	int geomnum;
	aten::ShapeParameter* shapes;

	aten::MaterialParameter* mtrls;

	int lightnum;
	aten::LightParameter* lights;
};

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
	const aten::ray* r,
	aten::hitrecord* rec,
	const Context* ctx)
{
	bool isHit = false;

	aten::hitrecord tmp;

	for (int i = 0; i < ctx->geomnum; i++) {
		const auto& s = ctx->shapes[i];
		if (AT_NAME::sphere::hit(s, *r, AT_MATH_EPSILON, AT_MATH_INF, tmp)) {
			if (tmp.t < rec->t) {
				*rec = tmp;
				rec->obj = (void*)&ctx->shapes[i];
				rec->mtrlid = ctx->shapes[i].mtrl.idx;

				isHit = true;
			}
		}
	}

	return isHit;
}

__global__ void raytracing(
	float4* p,
	int width, int height,
	Camera* camera,
	aten::ShapeParameter* shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	aten::LightParameter* lights, int lightnum)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	Context ctx;
	{
		ctx.geomnum = geomnum;
		ctx.shapes = shapes;
		ctx.mtrls = mtrls;
		ctx.lightnum = lightnum;
		ctx.lights = lights;
	}

	const auto idx = iy * camera->width + ix;

	float s = ix / (float)camera->width;
	float t = iy / (float)camera->height;

	CameraSampleResult camsample;
	sampleCamera(&camsample, camera, s, t);

	aten::vec3 contrib(0);
	aten::vec3 throughput(1);

	int depth = 0;

	aten::ray ray = camsample.r;

	while (depth < 5) {
		aten::hitrecord rec;

		if (intersect(&ray, &rec, &ctx)) {
			const aten::MaterialParameter& m = ctx.mtrls[rec.mtrlid];

			if (m.attrib.isEmissive) {
				auto emit = m.baseColor;
				contrib = throughput * emit;
				break;
			}

			// 交差位置の法線.
			// 物体からのレイの入出を考慮.
			const aten::vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

			if (m.attrib.isSingular || m.attrib.isTranslucent) {
				auto sampling = sampleMaterial(
					m,
					orienting_normal, 
					ray.dir,
					rec,
					nullptr,
					rec.u, rec.v);

				auto nextDir = normalize(sampling.dir);
				auto bsdf = sampling.bsdf;

				throughput *= bsdf;

				// Make next ray.
				ray = aten::ray(rec.p, nextDir);
			}
			else {
				for (int i = 0; i < lightnum; i++) {
					// TODO
					auto light = lights[i];
					auto* sphere = &ctx.shapes[light.object.idx];
					light.object.ptr = sphere;

					aten::LightSampleResult sampleres = sampleLight(light, rec.p, nullptr);

					aten::vec3 dirToLight = sampleres.dir;
					auto len = dirToLight.length();

					dirToLight.normalize();

					auto albedo = m.baseColor;

					aten::ray shadowRay(rec.p, dirToLight);

					aten::hitrecord tmpRec;

					auto funcHitTest = [&] AT_DEVICE_API(const aten::ray& _r, float t_min, float t_max, aten::hitrecord& _rec)
					{
						return intersect(&_r, &_rec, &ctx);
					};

					if (AT_NAME::scene::hitLight(funcHitTest, light, sampleres.pos, shadowRay, AT_MATH_EPSILON, AT_MATH_INF, tmpRec)) {
						auto lightColor = sampleres.finalColor;

						if (light.attrib.isInfinite) {
							len = 1.0f;
						}

						const auto c0 = max(0.0f, dot(orienting_normal, dirToLight));
						float c1 = 1.0f;

						if (!light.attrib.isSingular) {
							c1 = max(0.0f, dot(sampleres.nml, -dirToLight));
						}

						auto G = c0 * c1 / (len * len);

						contrib += throughput * (albedo * lightColor) * G;
					}
				}
				break;
			}
		}
		else {
			break;
		}

		depth++;
	}

	p[idx] = make_float4(contrib.x, contrib.y, contrib.z, 1);
}

__global__ void addFuncs()
{
	addLighFuncs();
	addMaterialFuncs();
}

void prepareRayTracing()
{
	addFuncs << <1, 1 >> > ();
}

void renderRayTracing(
	aten::vec4* image,
	int width, int height,
	const std::vector<aten::ShapeParameter>& shapes,
	const std::vector<aten::MaterialParameter>& mtrls,
	const std::vector<aten::LightParameter>& lights)
{
	Camera camera;
	initCamera(
		camera,
#if 0
		aten::vec3(0, 0, 0),
		aten::vec3(0, 0, -1),
#else
		aten::vec3(50.0, 52.0, 295.6),
		aten::vec3(50.0, 40.8, 119.0),
#endif
		aten::vec3(0, 1, 0),
		30,
		width, height);

#if 1
	aten::CudaMemory dst(sizeof(float4) * width * height);

	aten::TypedCudaMemory<Camera> cam(&camera, 1);
	
	aten::TypedCudaMemory<aten::ShapeParameter> shapeparam(shapes.size());
	shapeparam.writeByNum(&shapes[0], shapes.size());

	aten::TypedCudaMemory<aten::MaterialParameter> mtrlparam(mtrls.size());
	mtrlparam.writeByNum(&mtrls[0], mtrls.size());

	aten::TypedCudaMemory<aten::LightParameter> lightparam(lights.size());
	lightparam.writeByNum(&lights[0], lights.size());

	dim3 block(32, 32);
	dim3 grid(
		(width + block.x - 1) / block.x,
		(height + block.y - 1) / block.y);

	raytracing << <grid, block >> > (
	//raytracing << <dim3(1, 1), block >> > (
		(float4*)dst.ptr(), 
		width, height, 
		cam.ptr(),
		shapeparam.ptr(), shapeparam.num(),
		mtrlparam.ptr(),
		lightparam.ptr(), lightparam.num());

	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		AT_PRINTF("Cuda Kernel Err [%s]\n", cudaGetErrorString(err));
	}

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
