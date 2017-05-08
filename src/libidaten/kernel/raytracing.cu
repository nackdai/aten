#include "kernel/raytracing.h"
#include "kernel/context.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/intersect.cuh"
#include "kernel/bvh.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

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

struct Path {
	aten::ray ray;
	aten::vec3 throughput;
	aten::hitrecord rec;
	bool isHit;
	bool isTerminate;
};

__global__ void genPath(
	Path* paths,
	int width, int height,
	aten::CameraParameter* camera)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = iy * camera->width + ix;

	float s = ix / (float)camera->width;
	float t = iy / (float)camera->height;

	auto camsample = AT_NAME::PinholeCamera::sample(*camera, s, t, nullptr);

	auto& path = paths[idx];

	path.ray = camsample.r;
	path.throughput = aten::vec3(1);
	path.isHit = false;
	path.isTerminate = false;
}

__global__ void hitTest(
	Path* paths,
	int width, int height,
	aten::ShapeParameter* shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	aten::LightParameter* lights, int lightnum,
	aten::BVHNode* nodes)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = iy * width + ix;

	auto& path = paths[idx];
	path.isHit = false;

	if (path.isTerminate) {
		return;
	}

	Context ctx;
	{
		ctx.geomnum = geomnum;
		ctx.shapes = shapes;
		ctx.mtrls = mtrls;
		ctx.lightnum = lightnum;
		ctx.lights = lights;
		ctx.nodes = nodes;
	}
	
	aten::hitrecord rec;
	bool isHit = intersectBVH(ctx, path.ray, AT_MATH_EPSILON, AT_MATH_INF, rec);

	path.isHit = isHit;
	path.rec = rec;
}

__global__ void raytracing(
	float4* p,
	Path* paths,
	int width, int height,
	aten::ShapeParameter* shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	aten::LightParameter* lights, int lightnum,
	aten::BVHNode* nodes)
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
		ctx.nodes = nodes;
	}

	const auto idx = iy * width + ix;

	auto& path = paths[idx];

	if (!path.isHit) {
		return;
	}
	if (path.isTerminate) {
		return;
	}

	aten::vec3 contrib(0);

	const aten::MaterialParameter& m = ctx.mtrls[path.rec.mtrlid];

	if (m.attrib.isEmissive) {
		auto emit = m.baseColor;
		contrib = path.throughput * emit;

		path.isTerminate = true;
		p[idx] = make_float4(contrib.x, contrib.y, contrib.z, 1);
		
		return;
	}

	// 交差位置の法線.
	// 物体からのレイの入出を考慮.
	const aten::vec3 orienting_normal = dot(path.rec.normal, path.ray.dir) < 0.0 ? path.rec.normal : -path.rec.normal;

	if (m.attrib.isSingular || m.attrib.isTranslucent) {
		auto sampling = sampleMaterial(
			m,
			orienting_normal, 
			path.ray.dir,
			path.rec,
			nullptr,
			path.rec.u, path.rec.v);

		auto nextDir = normalize(sampling.dir);
		auto bsdf = sampling.bsdf;

		path.throughput *= bsdf;

		// Make next ray.
		path.ray = aten::ray(path.rec.p, nextDir);
	}
	else {
		// TODO
		auto light = lights[0];
		auto* sphere = &ctx.shapes[light.object.idx];
		light.object.ptr = sphere;

		aten::LightSampleResult sampleres = sampleLight(light, path.rec.p, nullptr);

		aten::vec3 dirToLight = sampleres.dir;
		auto len = dirToLight.length();

		dirToLight.normalize();

		auto albedo = m.baseColor;

		aten::ray shadowRay(path.rec.p, dirToLight);

		aten::hitrecord tmpRec;

		auto funcHitTest = [&] AT_DEVICE_API(const aten::ray& _r, float t_min, float t_max, aten::hitrecord& _rec)
		{
			//return intersect(&_r, &_rec, &ctx);
			return intersectBVH(ctx, _r, t_min, t_max, _rec);
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

			contrib += path.throughput * (albedo * lightColor) * G;
		}

		path.isTerminate = true;
		p[idx] = make_float4(contrib.x, contrib.y, contrib.z, 1);
	}
}

__global__ void addFuncs()
{
	addLighFuncs();
	addMaterialFuncs();
	addIntersectFuncs();
}

namespace idaten {
	void RayTracing::prepare()
	{
		addFuncs << <1, 1 >> > ();
	}

	void RayTracing::update(
		int width, int height,
		const aten::CameraParameter& camera,
		const std::vector<aten::ShapeParameter>& shapes,
		const std::vector<aten::MaterialParameter>& mtrls,
		const std::vector<aten::LightParameter>& lights,
		const std::vector<aten::BVHNode>& nodes)
	{
#if 0
		size_t size_stack = 0;
		checkCudaErrors(cudaThreadGetLimit(&size_stack, cudaLimitStackSize));
		checkCudaErrors(cudaThreadSetLimit(cudaLimitStackSize, 12928));
		checkCudaErrors(cudaThreadGetLimit(&size_stack, cudaLimitStackSize));

		AT_PRINTF("Stack size %d\n", size_stack);
#endif

		dst.init(sizeof(float4) * width * height);

		cam.init(sizeof(camera));
		cam.writeByNum(&camera, 1);

		shapeparam.init(shapes.size());
		shapeparam.writeByNum(&shapes[0], shapes.size());

		mtrlparam.init(mtrls.size());
		mtrlparam.writeByNum(&mtrls[0], mtrls.size());

		lightparam.init(lights.size());
		lightparam.writeByNum(&lights[0], lights.size());

		nodeparam.init(nodes.size());
		nodeparam.writeByNum(&nodes[0], nodes.size());
	}

	void RayTracing::render(
		aten::vec4* image,
		int width, int height)
	{
		dim3 block(32, 32);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		int depth = 0;

		aten::TypedCudaMemory<Path> paths;
		paths.init(width * height);

		genPath << <grid, block >> > (
			paths.ptr(),
			width, height,
			cam.ptr());

		//checkCudaErrors(cudaDeviceSynchronize());

		while (depth < 5) {
			hitTest << <grid, block >> > (
				paths.ptr(),
				width, height,
				shapeparam.ptr(), shapeparam.num(),
				mtrlparam.ptr(),
				lightparam.ptr(), lightparam.num(),
				nodeparam.ptr());

			auto err = cudaGetLastError();
			if (err != cudaSuccess) {
				AT_PRINTF("Cuda Kernel Err(hitTest) [%s]\n", cudaGetErrorString(err));
			}

			//checkCudaErrors(cudaDeviceSynchronize());

			raytracing << <grid, block >> > (
				(float4*)dst.ptr(),
				paths.ptr(),
				width, height,
				shapeparam.ptr(), shapeparam.num(),
				mtrlparam.ptr(),
				lightparam.ptr(), lightparam.num(),
				nodeparam.ptr());

			err = cudaGetLastError();
			if (err != cudaSuccess) {
				AT_PRINTF("Cuda Kernel Err(raytracing) [%s]\n", cudaGetErrorString(err));
			}

			//checkCudaErrors(cudaDeviceSynchronize());

			depth++;
		}

		checkCudaErrors(cudaDeviceSynchronize());

		dst.read(image, sizeof(aten::vec4) * width * height);
	}
}