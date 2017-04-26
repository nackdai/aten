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

__global__ void raytracing(
	float4* p,
	int width, int height,
	aten::CameraParameter* camera,
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

	const auto idx = iy * camera->width + ix;

	float s = ix / (float)camera->width;
	float t = iy / (float)camera->height;

	auto camsample = AT_NAME::PinholeCamera::sample(*camera, s, t, nullptr);

	aten::vec3 contrib(0);
	aten::vec3 throughput(1);

	int depth = 0;

	aten::ray ray = camsample.r;

	while (depth < 5) {
		aten::hitrecord rec;

		//if (intersect(&ray, &rec, &ctx)) {
		if (intersectBVH(ctx, ray, AT_MATH_EPSILON, AT_MATH_INF, rec)) {
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
	addIntersectFuncs();
}

void prepareRayTracing()
{
	addFuncs << <1, 1 >> > ();
}

void renderRayTracing(
	aten::vec4* image,
	int width, int height,
	const aten::CameraParameter& camera,
	const std::vector<aten::ShapeParameter>& shapes,
	const std::vector<aten::MaterialParameter>& mtrls,
	const std::vector<aten::LightParameter>& lights,
	const std::vector<aten::BVHNode>& nodes)
{
	aten::CudaMemory dst(sizeof(float4) * width * height);

	aten::TypedCudaMemory<aten::CameraParameter> cam(&camera, 1);
	
	aten::TypedCudaMemory<aten::ShapeParameter> shapeparam(shapes.size());
	shapeparam.writeByNum(&shapes[0], shapes.size());

	aten::TypedCudaMemory<aten::MaterialParameter> mtrlparam(mtrls.size());
	mtrlparam.writeByNum(&mtrls[0], mtrls.size());

	aten::TypedCudaMemory<aten::LightParameter> lightparam(lights.size());
	lightparam.writeByNum(&lights[0], lights.size());

	aten::TypedCudaMemory<aten::BVHNode> nodeparam(nodes.size());
	nodeparam.writeByNum(&nodes[0], nodes.size());

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
		lightparam.ptr(), lightparam.num(),
		nodeparam.ptr());

	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		AT_PRINTF("Cuda Kernel Err [%s]\n", cudaGetErrorString(err));
	}

	checkCudaErrors(cudaDeviceSynchronize());

	dst.read(image, sizeof(aten::vec4) * width * height);
}
