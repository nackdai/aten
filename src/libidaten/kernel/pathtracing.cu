#include "kernel/pathtracing.h"
#include "kernel/context.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/intersect.cuh"
#include "kernel/bvh.cuh"
#include "kernel/common.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

struct ShadowRay : public aten::ray {
	aten::vec3 lightcontrib;
	real distToLight;
	int targetLightId;

	struct {
		uint32_t isActive : 1;
	};
};

struct Path {
	aten::vec3 throughput;
	aten::vec3 contrib;
	aten::hitrecord rec;
	aten::sampler sampler;
	
	int mtrlid;

	real pdfb;

	bool isHit;
	bool isTerminate;
};

#define BLOCK_SIZE	(16)
#define BLOCK_SIZE2	(BLOCK_SIZE * BLOCK_SIZE)

inline AT_DEVICE_API int getIdx(int ix, int iy, int width)
{
	int X = ix / BLOCK_SIZE;
	int Y = iy / BLOCK_SIZE;

	//int base = Y * BLOCK_SIZE2 * (width / BLOCK_SIZE) + X * BLOCK_SIZE2;

	int XB = X * BLOCK_SIZE;
	int YB = Y * BLOCK_SIZE;

	int base = YB * width + XB * BLOCK_SIZE;

	const auto idx = base + (iy - YB) * BLOCK_SIZE + (ix - XB);

	return idx;
}

__global__ void genPath(
	Path* paths,
	aten::ray* rays,
	int width, int height,
	int sample, int maxSamples,
	int seed,
	aten::CameraParameter* camera)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	auto& path = paths[idx];
	path.sampler.init((iy * height * 4 + ix * 4) * maxSamples + sample + 1 + seed);

	float s = (ix + path.sampler.nextSample()) / (float)(camera->width);
	float t = (iy + path.sampler.nextSample()) / (float)(camera->height);

	AT_NAME::CameraSampleResult camsample;
	AT_NAME::PinholeCamera::sample(&camsample, camera, s, t);

	rays[idx] = camsample.r;

	path.throughput = aten::make_float3(1);
	path.mtrlid = -1;
	path.pdfb = 0.0f;
	path.isHit = false;
	path.isTerminate = false;

	// Accumulate value, so do not reset.
	//path.contrib = aten::make_float3(0);
}

__global__ void hitTest(
	Path* paths,
	aten::ray* rays,
	int width, int height,
	aten::ShapeParameter* shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	aten::LightParameter* lights, int lightnum,
	cudaTextureObject_t* nodes,
	aten::PrimitiveParamter* prims,
	cudaTextureObject_t vertices,
	aten::mat4* matrices)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	auto& path = paths[idx];
	path.isHit = false;

	if (path.isTerminate) {
		return;
	}

	Context ctxt;
	{
		ctxt.geomnum = geomnum;
		ctxt.shapes = shapes;
		ctxt.mtrls = mtrls;
		ctxt.lightnum = lightnum;
		ctxt.lights = lights;
		ctxt.nodes = nodes;
		ctxt.prims = prims;
		ctxt.vertices = vertices;
		ctxt.matrices = matrices;
	}
	
	aten::hitrecord rec;
	float t = AT_MATH_INF;
	bool isHit = intersectBVH(&ctxt, rays[idx], AT_MATH_EPSILON, AT_MATH_INF, &rec, t);

	path.isHit = isHit;
	path.rec = rec;
}

__global__ void shadeMiss(
	cudaSurfaceObject_t outSurface,
	Path* paths,
	int width, int height)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	auto& path = paths[idx];

	if (!path.isTerminate && !path.isHit) {
		// TODO
		path.contrib = aten::make_float3(0);
		path.isTerminate = true;
	}
}

__global__ void shade(
	cudaSurfaceObject_t outSurface,
	Path* paths,
	aten::ray* rays,
	ShadowRay* shadowRays,
	int width, int height,
	int depth, int rrDepth,
	aten::ShapeParameter* shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	aten::LightParameter* lights, int lightnum,
	cudaTextureObject_t* nodes,
	aten::PrimitiveParamter* prims,
	cudaTextureObject_t vertices,
	aten::mat4* matrices)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	Context ctxt;
	{
		ctxt.geomnum = geomnum;
		ctxt.shapes = shapes;
		ctxt.mtrls = mtrls;
		ctxt.lightnum = lightnum;
		ctxt.lights = lights;
		ctxt.nodes = nodes;
		ctxt.prims = prims;
		ctxt.vertices = vertices;
		ctxt.matrices = matrices;
	}

	const auto idx = getIdx(ix, iy, width);

	auto& path = paths[idx];
	const auto& ray = rays[idx];

	shadowRays[idx].isActive = false;

	if (!path.isHit) {
		return;
	}
	if (path.isTerminate) {
		return;
	}

	aten::MaterialParameter* mtrl = &ctxt.mtrls[path.rec.mtrlid];
	aten::MaterialParameter* prevMtrl = (path.mtrlid >= 0 ? &ctxt.mtrls[path.mtrlid] : nullptr);

	// 交差位置の法線.
	// 物体からのレイの入出を考慮.
	const aten::vec3 orienting_normal = dot(path.rec.normal, ray.dir) < 0.0 ? path.rec.normal : -path.rec.normal;

	// TODO
	// Apply normal map.

	// Implicit conection to light.
	if (mtrl->attrib.isEmissive) {
		if (depth == 0) {
			// Ray hits the light directly.
			path.contrib = mtrl->baseColor;
			path.isTerminate = true;
			return;
		}
		else if (prevMtrl && prevMtrl->attrib.isSingular) {
			auto emit = prevMtrl->baseColor;
			path.contrib += path.throughput * emit;
			path.isTerminate = true;
			return;
		}
		else {
			auto cosLight = dot(orienting_normal, -ray.dir);
			auto dist2 = (path.rec.p - ray.org).squared_length();

			if (cosLight >= 0) {
				auto pdfLight = 1 / path.rec.area;

				// Convert pdf area to sradian.
				// http://www.slideshare.net/h013/edubpt-v100
				// p31 - p35
				pdfLight = pdfLight * dist2 / cosLight;

				auto misW = path.pdfb / (pdfLight + path.pdfb);

				auto emit = mtrl->baseColor;

				path.contrib += path.throughput * misW * emit;

				// When ray hit the light, tracing will finish.
				path.isTerminate = true;
				return;
			}
		}
	}

	// Explicit conection to light.
	if (!mtrl->attrib.isSingular)
	{
		real lightSelectPdf = 1;
		aten::LightSampleResult sampleres;

		// TODO
		int lightidx = aten::cmpMin<int>(path.sampler.nextSample() * lightnum, lightnum - 1);
		lightSelectPdf = 1.0f / lightnum;

		auto light = ctxt.lights[lightidx];
		if (light.object.idx >= 0) {
			light.object.ptr = &ctxt.shapes[light.object.idx];
		}

		sampleLight(&sampleres, &ctxt, &light, path.rec.p, &path.sampler);

		const auto& posLight = sampleres.pos;
		const auto& nmlLight = sampleres.nml;
		real pdfLight = sampleres.pdf;

		auto lightobj = sampleres.obj;

		auto dirToLight = normalize(sampleres.dir);

		auto cosShadow = dot(orienting_normal, dirToLight);

		real pdfb = samplePDF(mtrl, orienting_normal, ray.dir, dirToLight, path.rec.u, path.rec.v);
		auto bsdf = sampleBSDF(mtrl, orienting_normal, ray.dir, dirToLight, path.rec.u, path.rec.v);

		bsdf *= path.throughput;

		// Get light color.
		auto emit = sampleres.finalColor;

		shadowRays[idx].org = path.rec.p;
		shadowRays[idx].dir = dirToLight;
		shadowRays[idx].lightcontrib = aten::make_float3(0);
		shadowRays[idx].distToLight = sampleres.dir.length();
		shadowRays[idx].targetLightId = lightidx;

		if (light.attrib.isSingular || light.attrib.isInfinite) {
			if (pdfLight > real(0)) {
				// TODO
				// ジオメトリタームの扱いについて.
				// singular light の場合は、finalColor に距離の除算が含まれている.
				// inifinite light の場合は、無限遠方になり、pdfLightに含まれる距離成分と打ち消しあう？.
				// （打ち消しあうので、pdfLightには距離成分は含んでいない）.
				auto misW = pdfLight / (pdfb + pdfLight);
				shadowRays[idx].lightcontrib = (misW * bsdf * emit * cosShadow / pdfLight) / lightSelectPdf;
				shadowRays[idx].isActive = true;
			}
		}
		else {
			auto cosLight = dot(nmlLight, -dirToLight);

			if (cosShadow >= 0 && cosLight >= 0) {
				auto dist2 = sampleres.dir.squared_length();
				auto G = cosShadow * cosLight / dist2;

				if (pdfb > real(0) && pdfLight > real(0)) {
					// Convert pdf from steradian to area.
					// http://www.slideshare.net/h013/edubpt-v100
					// p31 - p35
					pdfb = pdfb * cosLight / dist2;

					auto misW = pdfLight / (pdfb + pdfLight);

					shadowRays[idx].lightcontrib = (misW * (bsdf * emit * G) / pdfLight) / lightSelectPdf;
					shadowRays[idx].isActive = true;
				}
			}
		}
	}

	real russianProb = real(1);

	if (depth > rrDepth) {
		auto t = normalize(path.throughput);
		auto p = aten::cmpMax(t.r, aten::cmpMax(t.g, t.b));

		russianProb = path.sampler.nextSample();

		if (russianProb >= p) {
			path.contrib = aten::make_float3(0);
			path.isTerminate = true;
		}
		else {
			russianProb = p;
		}
	}
			
	AT_NAME::MaterialSampling sampling;

	sampleMaterial(
		&sampling,
		mtrl,
		orienting_normal,
		ray.dir,
		path.rec.normal,
		&path.sampler,
		path.rec.u, path.rec.v);

	auto nextDir = normalize(sampling.dir);
	auto pdfb = sampling.pdf;
	auto bsdf = sampling.bsdf;

	real c = 1;
	if (!mtrl->attrib.isSingular) {
		// TODO
		// AMDのはabsしているが....
		//c = aten::abs(dot(orienting_normal, nextDir));
		c = dot(orienting_normal, nextDir);
	}

	if (pdfb > 0 && c > 0) {
		path.throughput *= bsdf * c / pdfb;
		path.throughput /= russianProb;
	}
	else {
		path.isTerminate = true;
	}

	// Make next ray.
	rays[idx] = aten::ray(path.rec.p, nextDir);

	path.mtrlid = path.rec.mtrlid;
	path.pdfb = pdfb;
}

__global__ void hitShadowRay(
	Path* paths,
	ShadowRay* shadowRays,
	int width, int height,
	aten::ShapeParameter* shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	aten::LightParameter* lights, int lightnum,
	cudaTextureObject_t* nodes,
	aten::PrimitiveParamter* prims,
	cudaTextureObject_t vertices,
	aten::mat4* matrices)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	Context ctxt;
	{
		ctxt.geomnum = geomnum;
		ctxt.shapes = shapes;
		ctxt.mtrls = mtrls;
		ctxt.lightnum = lightnum;
		ctxt.lights = lights;
		ctxt.nodes = nodes;
		ctxt.prims = prims;
		ctxt.vertices = vertices;
		ctxt.matrices = matrices;
	}

	const auto idx = getIdx(ix, iy, width);

	auto& shadowRay = shadowRays[idx];

	if (shadowRay.isActive) {
		auto& path = paths[idx];

		aten::hitrecord rec;
		float t = AT_MATH_INF;
		bool isHit = intersectBVH(&ctxt, shadowRay, AT_MATH_EPSILON, AT_MATH_INF, &rec, t);

		auto light = ctxt.lights[shadowRay.targetLightId];
		if (light.object.idx >= 0) {
			light.object.ptr = &ctxt.shapes[light.object.idx];
		}

		real distHitObjToRayOrg = (rec.p - shadowRay.org).length();

		auto obj = &ctxt.shapes[rec.objid];
		
		shadowRay.isActive = AT_NAME::scene::hitLight(
			isHit, 
			light.attrib,
			light.object.ptr,
			shadowRay.distToLight,
			distHitObjToRayOrg,
			t,
			obj);

		if (shadowRay.isActive) {
			path.contrib += shadowRay.lightcontrib;
		}
	}
}

__global__ void gather(
	cudaSurfaceObject_t outSurface,
	Path* paths,
	int width, int height,
	int sample)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	const auto& path = paths[idx];

	float4 data;
	surf2Dread(&data, outSurface, ix * sizeof(float4), iy);

	// First data.w value is 0.
	int n = data.w;
	data = n * data + make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 0) / sample;
	data /= (n + 1);
	data.w = n + 1;

	surf2Dwrite(
		data,
		outSurface,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}

namespace idaten {
	void PathTracing::prepare()
	{
		addFuncs();
	}

#include "misc/timer.h"
	aten::SystemTime getSystemTime()
	{
		SYSTEMTIME time;
		::GetSystemTime(&time);

		aten::SystemTime ret;
		ret.year = time.wYear;
		ret.month = time.wMonth;
		ret.dayOfWeek = time.wDayOfWeek;
		ret.day = time.wDay;
		ret.hour = time.wHour;
		ret.minute = time.wMinute;
		ret.second = time.wSecond;
		ret.milliSeconds = time.wMilliseconds;

		return std::move(ret);
	}

	static bool doneSetStackSize = false;

	void PathTracing::render(
		aten::vec4* image,
		int width, int height)
	{
#ifdef __AT_DEBUG__
		if (!doneSetStackSize) {
			size_t val = 0;
			cudaThreadGetLimit(&val, cudaLimitStackSize);
			cudaThreadSetLimit(cudaLimitStackSize, val * 2);
			doneSetStackSize = true;
		}
#endif

		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		int depth = 0;

		idaten::TypedCudaMemory<Path> paths;
		paths.init(width * height);

		idaten::TypedCudaMemory<aten::ray> rays;
		rays.init(width * height);

		idaten::TypedCudaMemory<ShadowRay> shadowRays;
		shadowRays.init(width * height);

		CudaGLResourceMap rscmap(&glimg);
		auto outputSurf = glimg.bind();

		auto vtxTex = vtxparams.bind();

		std::vector<cudaTextureObject_t> tmp;
		for (int i = 0; i < nodeparam.size(); i++) {
			auto nodeTex = nodeparam[i].bind();
			tmp.push_back(nodeTex);
		}
		nodetex.writeByNum(&tmp[0], tmp.size());

		static const int maxSamples = 1;
		static const int maxDepth = 5;
		static const int rrDepth = 3;

		auto time = getSystemTime();

		for (int i = 0; i < maxSamples; i++) {
#if 1
			genPath << <grid, block >> > (
			//genPath << <1, 1 >> > (
				paths.ptr(),
				rays.ptr(),
				width, height,
				i, maxSamples,
				time.milliSeconds,
				cam.ptr());

			depth = 0;

			while (depth < maxDepth) {
				hitTest << <grid, block >> > (
				//hitTest << <1, 1 >> > (
					paths.ptr(),
					rays.ptr(),
					width, height,
					shapeparam.ptr(), shapeparam.num(),
					mtrlparam.ptr(),
					lightparam.ptr(), lightparam.num(),
					nodetex.ptr(),
					primparams.ptr(),
					vtxTex,
					mtxparams.ptr());

				auto err = cudaGetLastError();
				if (err != cudaSuccess) {
					AT_PRINTF("Cuda Kernel Err(hitTest) [%s]\n", cudaGetErrorString(err));
				}

				shadeMiss << <grid, block >> > (
				//shadeMiss << <1, 1 >> > (
					outputSurf,
					paths.ptr(),
					width, height);

				shade << <grid, block >> > (
				//shade << <1, 1 >> > (
					outputSurf,
					paths.ptr(),
					rays.ptr(),
					shadowRays.ptr(),
					width, height,
					depth, rrDepth,
					shapeparam.ptr(), shapeparam.num(),
					mtrlparam.ptr(),
					lightparam.ptr(), lightparam.num(),
					nodetex.ptr(),
					primparams.ptr(),
					vtxTex,
					mtxparams.ptr());

				err = cudaGetLastError();
				if (err != cudaSuccess) {
					AT_PRINTF("Cuda Kernel Err(shade) [%s]\n", cudaGetErrorString(err));
				}

				hitShadowRay << <grid, block >> > (
				//hitShadowRay << <1, 1 >> > (
					paths.ptr(),
					shadowRays.ptr(),
					width, height,
					shapeparam.ptr(), shapeparam.num(),
					mtrlparam.ptr(),
					lightparam.ptr(), lightparam.num(),
					nodetex.ptr(),
					primparams.ptr(),
					vtxTex,
					mtxparams.ptr());

				err = cudaGetLastError();
				if (err != cudaSuccess) {
					AT_PRINTF("Cuda Kernel Err(hitShadowRay) [%s]\n", cudaGetErrorString(err));
				}

				depth++;
			}
#endif
		}

		gather << <grid, block >> > (
		//gather << <1, 1 >> > (
			outputSurf,
			paths.ptr(),
			width, height,
			maxSamples);

		vtxparams.unbind();
		for (int i = 0; i < nodeparam.size(); i++) {
			nodeparam[i].unbind();
		}
		nodetex.reset();

		//dst.read(image, sizeof(aten::vec4) * width * height);
	}
}
