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

struct ray : public aten::ray {
	AT_DEVICE_API ray() : aten::ray()
	{
		isActive = true;
	}
	AT_DEVICE_API ray(const aten::vec3& o, const aten::vec3& d) : aten::ray(o, d)
	{
		isActive = true;
	}

	struct {
		uint32_t isActive : 1;
	};
};

struct Path {
	aten::ray ray;
	aten::vec3 throughput;
	aten::vec3 contrib;
	aten::hitrecord rec;
	aten::sampler sampler;
	aten::MaterialParameter* mtrl;

	aten::vec3 lightPos;
	aten::vec3 lightcontrib;
	aten::LightParameter* targetLight;

	real pdfb;

	bool isHit;
	bool isTerminate;
};

__global__ void genPath(
	Path* paths,
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

	const auto idx = iy * camera->width + ix;

	auto& path = paths[idx];
	path.sampler.init((iy * height * 4 + ix * 4) * maxSamples + sample + 1 + seed);

	float s = (ix + path.sampler.nextSample()) / (float)(camera->width);
	float t = (iy + path.sampler.nextSample()) / (float)(camera->height);

	AT_NAME::CameraSampleResult camsample;
	AT_NAME::PinholeCamera::sample(&camsample, camera, s, t);

	path.ray = camsample.r;
	path.throughput = aten::make_float3(1);
	path.contrib = aten::make_float3(0);
	path.mtrl = nullptr;
	path.lightcontrib = aten::make_float3(0);
	path.targetLight = nullptr;
	path.pdfb = 0.0f;
	path.isHit = false;
	path.isTerminate = false;
}

__global__ void hitTest(
	Path* paths,
	int width, int height,
	aten::ShapeParameter* shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	aten::LightParameter* lights, int lightnum,
	cudaTextureObject_t* nodes,
	aten::PrimitiveParamter* prims,
	cudaTextureObject_t vertices)
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
	}
	
	aten::hitrecord rec;
	bool isHit = intersectBVH(&ctxt, path.ray, AT_MATH_EPSILON, AT_MATH_INF, &rec);

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

	const auto idx = iy * width + ix;

	auto& path = paths[idx];

	if (!path.isTerminate && !path.isHit) {
		surf2Dwrite(
			make_float4(0, 0, 0, 1),
			outSurface, 
			ix * sizeof(float4), iy, 
			cudaBoundaryModeTrap);

		path.isTerminate = true;
	}
}

__global__ void shade(
	cudaSurfaceObject_t outSurface,
	Path* paths,
	ray* shadowRays,
	int width, int height,
	int depth, int rrDepth,
	aten::ShapeParameter* shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	aten::LightParameter* lights, int lightnum,
	cudaTextureObject_t* nodes,
	aten::PrimitiveParamter* prims,
	cudaTextureObject_t vertices)
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
	}

	const auto idx = iy * width + ix;

	auto& path = paths[idx];

	if (!path.isHit) {
		return;
	}
	if (path.isTerminate) {
		return;
	}

	aten::MaterialParameter* mtrl = &ctxt.mtrls[path.rec.mtrlid];

	// 交差位置の法線.
	// 物体からのレイの入出を考慮.
	const aten::vec3 orienting_normal = dot(path.rec.normal, path.ray.dir) < 0.0 ? path.rec.normal : -path.rec.normal;

	// TODO
	// Apply normal map.

	bool willContinue = true;

	// Implicit conection to light.
	if (mtrl->attrib.isEmissive) {
		if (depth == 0) {
			// Ray hits the light directly.
			path.contrib = mtrl->baseColor;
			path.isTerminate = true;
			willContinue = false;
		}
		else if (path.mtrl && path.mtrl->attrib.isSingular) {
			auto emit = path.mtrl->baseColor;
			path.contrib += path.throughput * emit;
			willContinue = false;
		}
		else {
			auto cosLight = dot(orienting_normal, -path.ray.dir);
			auto dist2 = (path.rec.p - path.ray.org).squared_length();

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
				willContinue = false;
			}
		}
	}

	if (!willContinue) {
		surf2Dwrite(
			make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 1),
			outSurface,
			ix * sizeof(float4), iy,
			cudaBoundaryModeTrap);

		path.isTerminate = true;
		return;
	}

	// Explicit conection to light.
	if (!mtrl->attrib.isSingular)
	{
		real lightSelectPdf = 1;
		aten::LightSampleResult sampleres;

		// TODO
		int lightidx = aten::cmpMin<int>(path.sampler.nextSample() * lightnum, lightnum - 1);
		auto light = &ctxt.lights[lightidx];

		if (light) {
			const auto& posLight = sampleres.pos;
			const auto& nmlLight = sampleres.nml;
			real pdfLight = sampleres.pdf;

			auto lightobj = sampleres.obj;

			auto dirToLight = normalize(sampleres.dir);
			shadowRays[idx] = ray(path.rec.p, dirToLight);

			auto cosShadow = dot(orienting_normal, dirToLight);

			real pdfb = samplePDF(mtrl, orienting_normal, path.ray.dir, dirToLight, path.rec.u, path.rec.v);
			auto bsdf = sampleBSDF(mtrl, orienting_normal, path.ray.dir, dirToLight, path.rec.u, path.rec.v);

			bsdf *= path.throughput;

			// Get light color.
			auto emit = sampleres.finalColor;

			path.lightPos = posLight;
			path.targetLight = light;
			path.lightcontrib = aten::make_float3(0);

			if (light->attrib.isSingular || light->attrib.isInfinite) {
				if (pdfLight > real(0)) {
					// TODO
					// ジオメトリタームの扱いについて.
					// singular light の場合は、finalColor に距離の除算が含まれている.
					// inifinite light の場合は、無限遠方になり、pdfLightに含まれる距離成分と打ち消しあう？.
					// （打ち消しあうので、pdfLightには距離成分は含んでいない）.
					auto misW = pdfLight / (pdfb + pdfLight);
					path.lightcontrib = (misW * bsdf * emit * cosShadow / pdfLight) / lightSelectPdf;
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

						path.lightcontrib = (misW * (bsdf * emit * G) / pdfLight) / lightSelectPdf;
					}
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
			willContinue = false;
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
		path.ray.dir,
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
		willContinue = false;
	}

	// Make next ray.
	path.ray = aten::ray(path.rec.p, nextDir);

	path.mtrl = mtrl;
	path.pdfb = pdfb;

	if (!willContinue) {
		path.isTerminate = true;
	}
}

__global__ void hitShadowRay(
	Path* paths,
	ray* shadowRays,
	int width, int height,
	aten::ShapeParameter* shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	aten::LightParameter* lights, int lightnum,
	cudaTextureObject_t* nodes,
	aten::PrimitiveParamter* prims,
	cudaTextureObject_t vertices)
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
	}

	const auto idx = iy * width + ix;

	auto& shadowRay = shadowRays[idx];

	if (shadowRay.isActive) {
		auto& path = paths[idx];

		aten::hitrecord rec;
		bool isHit = intersectBVH(&ctxt, shadowRay, AT_MATH_EPSILON, AT_MATH_INF, &rec);

		
		shadowRay.isActive = AT_NAME::scene::hitLight(
			isHit, 
			path.targetLight, 
			path.lightPos,
			shadowRay, 
			AT_MATH_EPSILON, AT_MATH_INF, 
			&rec);

		if (shadowRay.isActive) {
			path.contrib += path.lightcontrib;
		}
	}
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

	void PathTracing::render(
		aten::vec4* image,
		int width, int height)
	{
		dim3 block(16, 16);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		int depth = 0;

		idaten::TypedCudaMemory<Path> paths;
		paths.init(width * height);

		idaten::TypedCudaMemory<ray> shadowRays;
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

		static const int maxSamples = 5;
		static const int maxDepth = 5;
		static const int rrDepth = 3;

		auto time = getSystemTime();

		for (int i = 0; i < maxSamples; i++) {
			genPath << <grid, block >> > (
				paths.ptr(),
				width, height,
				i, maxSamples,
				time.milliSeconds,
				cam.ptr());

			while (depth < maxDepth) {
				hitTest << <grid, block >> > (
					paths.ptr(),
					width, height,
					shapeparam.ptr(), shapeparam.num(),
					mtrlparam.ptr(),
					lightparam.ptr(), lightparam.num(),
					nodetex.ptr(),
					primparams.ptr(),
					vtxTex);

				auto err = cudaGetLastError();
				if (err != cudaSuccess) {
					AT_PRINTF("Cuda Kernel Err(hitTest) [%s]\n", cudaGetErrorString(err));
				}

				shadeMiss << <grid, block >> > (
					outputSurf,
					paths.ptr(),
					width, height);

				shade << <grid, block >> > (
					outputSurf,
					paths.ptr(),
					shadowRays.ptr(),
					width, height,
					depth, rrDepth,
					shapeparam.ptr(), shapeparam.num(),
					mtrlparam.ptr(),
					lightparam.ptr(), lightparam.num(),
					nodetex.ptr(),
					primparams.ptr(),
					vtxTex);

				err = cudaGetLastError();
				if (err != cudaSuccess) {
					AT_PRINTF("Cuda Kernel Err(shade) [%s]\n", cudaGetErrorString(err));
				}

				hitShadowRay << <grid, block >> > (
					paths.ptr(),
					shadowRays.ptr(),
					width, height,
					shapeparam.ptr(), shapeparam.num(),
					mtrlparam.ptr(),
					lightparam.ptr(), lightparam.num(),
					nodetex.ptr(),
					primparams.ptr(),
					vtxTex);

				err = cudaGetLastError();
				if (err != cudaSuccess) {
					AT_PRINTF("Cuda Kernel Err(hitShadowRay) [%s]\n", cudaGetErrorString(err));
				}

				depth++;
			}

			checkCudaErrors(cudaDeviceSynchronize());
		}

		vtxparams.unbind();
		for (int i = 0; i < nodeparam.size(); i++) {
			nodeparam[i].unbind();
		}
		nodetex.reset();

		//dst.read(image, sizeof(aten::vec4) * width * height);
	}
}
