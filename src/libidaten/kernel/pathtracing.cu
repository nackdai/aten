#include "kernel/pathtracing.h"
#include "kernel/context.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/intersect.cuh"
#include "kernel/bvh.cuh"
#include "kernel/compaction.h"
#include "kernel/pt_common.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

__global__ void genPath(
	idaten::PathTracing::Path* paths,
	aten::ray* rays,
	int width, int height,
	int sample, int maxSamples,
	int seed,
	const aten::CameraParameter* __restrict__ camera)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	auto& path = paths[idx];
	path.isHit = false;

	if (path.isKill) {
		path.isTerminate = true;
		return;
	}

	path.sampler.init((iy * height * 4 + ix * 4) * maxSamples + sample + 1 + seed);

	float s = (ix + path.sampler.nextSample()) / (float)(camera->width);
	float t = (iy + path.sampler.nextSample()) / (float)(camera->height);

	AT_NAME::CameraSampleResult camsample;
	AT_NAME::PinholeCamera::sample(&camsample, camera, s, t);

	rays[idx] = camsample.r;

	path.throughput = aten::vec3(1);
	path.pdfb = 0.0f;
	path.isTerminate = false;
	path.isSingular = false;

	path.samples += 1;

	// Accumulate value, so do not reset.
	//path.contrib = aten::vec3(0);
}

// NOTE
// persistent thread.
// https://gist.github.com/guozhou/b972bb42bbc5cba1f062#file-persistent-cpp-L15

// NOTE
// compute capability 6.0
// http://homepages.math.uic.edu/~jan/mcs572/performance_considerations.pdf
// p3

#define NUM_SM				64	// no. of streaming multiprocessors
#define NUM_WARP_PER_SM		64	// maximum no. of resident warps per SM
#define NUM_BLOCK_PER_SM	32	// maximum no. of resident blocks per SM
#define NUM_BLOCK			(NUM_SM * NUM_BLOCK_PER_SM)
#define NUM_WARP_PER_BLOCK	(NUM_WARP_PER_SM / NUM_BLOCK_PER_SM)
#define WARP_SIZE			32

__device__ unsigned int headDev = 0;

#define ENABLE_PERSISTENT_THREAD

__global__ void hitTest(
	idaten::PathTracing::Path* paths,
	aten::Intersection* isects,
	aten::ray* rays,
	int* hitbools,
	int width, int height,
	const aten::ShapeParameter* __restrict__ shapes, int geomnum,
	const aten::MaterialParameter* __restrict__ mtrls,
	const aten::LightParameter* __restrict__ lights, int lightnum,
	cudaTextureObject_t* nodes,
	const aten::PrimitiveParamter* __restrict__ prims,
	cudaTextureObject_t vtxPos,
	aten::mat4* matrices)
{
#ifdef ENABLE_PERSISTENT_THREAD
	// warp-wise head index of tasks in a block
	__shared__ volatile unsigned int headBlock[NUM_WARP_PER_BLOCK];

	volatile unsigned int& headWarp = headBlock[threadIdx.y];

	if (blockIdx.x == 0 && threadIdx.x == 0) {
		headDev = 0;
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
		ctxt.vtxPos = vtxPos;
		ctxt.matrices = matrices;
	}

	do
	{
		// let lane 0 fetch [wh, wh + WARP_SIZE - 1] for a warp
		if (threadIdx.x == 0) {
			headWarp = atomicAdd(&headDev, WARP_SIZE);
		}
		// task index per thread in a warp
		unsigned int idx = headWarp + threadIdx.x;

		if (idx >= width * height) {
			return;
		}

		auto& path = paths[idx];
		path.isHit = false;

		hitbools[idx] = 0;

		if (path.isTerminate) {
			continue;
		}

		aten::Intersection isect;

		bool isHit = intersectBVH(&ctxt, rays[idx], &isect);

		isects[idx].t = isect.t;
		isects[idx].objid = isect.objid;
		isects[idx].mtrlid = isect.mtrlid;
		isects[idx].area = isect.area;
		isects[idx].primid = isect.primid;
		isects[idx].a = isect.a;
		isects[idx].b = isect.b;

		path.isHit = isHit;

		hitbools[idx] = isHit ? 1 : 0;
	} while (true);
#else
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= width * height) {
		return;
	}

	auto& path = paths[idx];
	path.isHit = false;

	hitbools[idx] = 0;

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
		ctxt.vtxPos = vtxPos;
		ctxt.matrices = matrices;
	}

	aten::Intersection isect;

	bool isHit = intersectBVH(&ctxt, rays[idx], &isect);

	isects[idx].t = isect.t;
	isects[idx].objid = isect.objid;
	isects[idx].mtrlid = isect.mtrlid;
	isects[idx].area = isect.area;
	isects[idx].primid = isect.primid;
	isects[idx].a = isect.a;
	isects[idx].b = isect.b;

	path.isHit = isHit;

	hitbools[idx] = isHit ? 1 : 0;
#endif
}

template <bool isFirstBounce>
__global__ void shadeMiss(
	idaten::PathTracing::Path* paths,
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
		auto bg = aten::vec3(0);

		path.contrib += path.throughput * bg;

		path.isTerminate = true;

		if (isFirstBounce) {
			path.isKill = true;
		}
	}
}

template <bool isFirstBounce>
__global__ void shadeMissWithEnvmap(
	cudaTextureObject_t* textures,
	int envmapIdx,
	real envmapAvgIllum,
	idaten::PathTracing::Path* paths,
	const aten::ray* __restrict__ rays,
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
		auto r = rays[idx];

		auto uv = AT_NAME::envmap::convertDirectionToUV(r.dir);

		auto bg = tex2D<float4>(textures[envmapIdx], uv.x, uv.y);
		auto emit = aten::vec3(bg.x, bg.y, bg.z);

		float misW = 1.0f;
		if (isFirstBounce) {
			path.isKill = true;
		}
		else {
			auto pdfLight = AT_NAME::ImageBasedLight::samplePdf(emit, envmapAvgIllum);
			misW = path.pdfb / (pdfLight + path.pdfb);
		}

		path.contrib += path.throughput * misW * emit;

		path.isTerminate = true;
	}
}

__global__ void shade(
	cudaSurfaceObject_t outSurface,
	idaten::PathTracing::Path* paths,
	int* hitindices,
	int hitnum,
	const aten::Intersection* __restrict__ isects,
	aten::ray* rays,
	int depth, int rrDepth,
	const aten::ShapeParameter* __restrict__ shapes, int geomnum,
	const aten::MaterialParameter* __restrict__ mtrls,
	const aten::LightParameter* __restrict__ lights, int lightnum,
	cudaTextureObject_t* nodes,
	const aten::PrimitiveParamter* __restrict__ prims,
	cudaTextureObject_t vtxPos,
	cudaTextureObject_t vtxNml,
	const aten::mat4* __restrict__ matrices,
	cudaTextureObject_t* textures)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= hitnum) {
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
		ctxt.vtxPos = vtxPos;
		ctxt.vtxNml = vtxNml;
		ctxt.matrices = matrices;
		ctxt.textures = textures;
	}

	idx = hitindices[idx];

	//idx = (480 - 360) * 640 + 345;
	//idx = getIdx(43, 5, 640);

	auto& path = paths[idx];
	const auto& ray = rays[idx];

	aten::hitrecord rec;

	const auto& isect = isects[idx];

	auto obj = &ctxt.shapes[isect.objid];
	evalHitResult(&ctxt, obj, ray, &rec, &isect);

	const aten::MaterialParameter* mtrl = &ctxt.mtrls[rec.mtrlid];

	// 交差位置の法線.
	// 物体からのレイの入出を考慮.
	const aten::vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

	// TODO
	// Apply normal map.

	// Implicit conection to light.
	if (mtrl->attrib.isEmissive) {
		float weight = 1.0f;

		if (depth > 0 && !path.isSingular) {
			auto cosLight = dot(orienting_normal, -ray.dir);
			auto dist2 = aten::squared_length(rec.p - ray.org);

			if (cosLight >= 0) {
				auto pdfLight = 1 / rec.area;

				// Convert pdf area to sradian.
				// http://www.slideshare.net/h013/edubpt-v100
				// p31 - p35
				pdfLight = pdfLight * dist2 / cosLight;

				weight = path.pdfb / (pdfLight + path.pdfb);
			}
		}

		path.contrib += path.throughput * weight * mtrl->baseColor;

		// When ray hit the light, tracing will finish.
		path.isTerminate = true;
		return;
	}

	// Explicit conection to light.
	if (!mtrl->attrib.isSingular)
	{
		real lightSelectPdf = 1;
		aten::LightSampleResult sampleres;

		// TODO
		// Importance sampling.
		int lightidx = aten::cmpMin<int>(path.sampler.nextSample() * lightnum, lightnum - 1);
		lightSelectPdf = 1.0f / lightnum;

		auto light = ctxt.lights[lightidx];

		sampleLight(&sampleres, &ctxt, &light, rec.p, orienting_normal, &path.sampler);

		const auto& posLight = sampleres.pos;
		const auto& nmlLight = sampleres.nml;
		real pdfLight = sampleres.pdf;

		auto lightobj = sampleres.obj;

		auto dirToLight = normalize(sampleres.dir);
		auto distToLight = length(posLight - rec.p);

		real distHitObjToRayOrg = AT_MATH_INF;

		// Ray aim to the area light.
		// So, if ray doesn't hit anything in intersectCloserBVH, ray hit the area light.
		auto hitobj = lightobj;

		aten::Intersection isectTmp;
		aten::ray shadowRay(rec.p, dirToLight);

		bool isHit = false;

#if 0
		if (light.type == aten::LightType::Area) {
			// Area.
			isHit = intersectCloserBVH(&ctxt, shadowRay, &isectTmp, distToLight - AT_MATH_EPSILON);
		}
		else if (light.attrib.isInfinite) {
			// IBL, Directional.
			isHit = intersectAnyBVH(&ctxt, shadowRay, &isectTmp);
		}
		else {
			// Point, Spot.
			isHit = intersectCloserBVH(&ctxt, shadowRay, &isectTmp, distToLight - AT_MATH_EPSILON);
		}
#else
		isHit = intersectCloserBVH(&ctxt, shadowRay, &isectTmp, distToLight - AT_MATH_EPSILON);
#endif

		if (isHit) {
			hitobj = (void*)&ctxt.shapes[isectTmp.objid];
		}

		isHit = AT_NAME::scene::hitLight(
			isHit,
			light.attrib,
			lightobj,
			distToLight,
			distHitObjToRayOrg,
			isectTmp.t,
			hitobj);

		if (isHit) {
			auto cosShadow = dot(orienting_normal, dirToLight);

			real pdfb = samplePDF(mtrl, orienting_normal, ray.dir, dirToLight, rec.u, rec.v);
			auto bsdf = sampleBSDF(mtrl, orienting_normal, ray.dir, dirToLight, rec.u, rec.v);

			bsdf *= path.throughput;

			// Get light color.
			auto emit = sampleres.finalColor;

			if (light.attrib.isSingular || light.attrib.isInfinite) {
				if (pdfLight > real(0) && cosShadow >= 0) {
					// TODO
					// ジオメトリタームの扱いについて.
					// singular light の場合は、finalColor に距離の除算が含まれている.
					// inifinite light の場合は、無限遠方になり、pdfLightに含まれる距離成分と打ち消しあう？.
					// （打ち消しあうので、pdfLightには距離成分は含んでいない）.
					auto misW = pdfLight / (pdfb + pdfLight);
					path.contrib += (misW * bsdf * emit * cosShadow / pdfLight) / lightSelectPdf;
				}
			}
			else {
				auto cosLight = dot(nmlLight, -dirToLight);

				if (cosShadow >= 0 && cosLight >= 0) {
					auto dist2 = aten::squared_length(sampleres.dir);
					auto G = cosShadow * cosLight / dist2;

					if (pdfb > real(0) && pdfLight > real(0)) {
						// Convert pdf from steradian to area.
						// http://www.slideshare.net/h013/edubpt-v100
						// p31 - p35
						pdfb = pdfb * cosLight / dist2;

						auto misW = pdfLight / (pdfb + pdfLight);

						path.contrib += (misW * (bsdf * emit * G) / pdfLight) / lightSelectPdf;
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
			//path.contrib = aten::vec3(0);
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
		rec.normal,
		&path.sampler,
		rec.u, rec.v);

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
	rays[idx] = aten::ray(rec.p, nextDir);

	path.pdfb = pdfb;
	path.isSingular = mtrl->attrib.isSingular;
}

__global__ void hitShadowRay(
	idaten::PathTracing::Path* paths,
	int* hitindices,
	int hitnum,
	idaten::PathTracing::ShadowRay* shadowRays,
	aten::ShapeParameter* shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	aten::LightParameter* lights, int lightnum,
	cudaTextureObject_t* nodes,
	aten::PrimitiveParamter* prims,
	cudaTextureObject_t vtxPos,
	aten::mat4* matrices)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= hitnum) {
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
		ctxt.vtxPos = vtxPos;
		ctxt.matrices = matrices;
	}

	idx = hitindices[idx];

	auto& shadowRay = shadowRays[idx];

	if (shadowRay.isActive) {
		auto light = &ctxt.lights[shadowRay.targetLightId];
		auto lightobj = (light->objid >= 0 ? &ctxt.shapes[light->objid] : nullptr);

		real distHitObjToRayOrg = AT_MATH_INF;

		// Ray aim to the area light.
		// So, if ray doesn't hit anything in intersectCloserBVH, ray hit the area light.
		const aten::ShapeParameter* hitobj = lightobj;

		aten::Intersection isect;

		bool isHit = false;

		if (light->type == aten::LightType::Area) {
			isHit = intersectCloserBVH(&ctxt, shadowRay, &isect, shadowRay.distToLight - AT_MATH_EPSILON);			
			//isHit = intersectBVH(&ctxt, shadowRay, &isect);

			if (isHit) {
				hitobj = &ctxt.shapes[isect.objid];
#if 0
				aten::hitrecord rec;
				evalHitResult(&ctxt, hitobj, shadowRay, &rec, &isect);

				distHitObjToRayOrg = (rec.p - shadowRay.org).length();
#endif
			}
		}
		else {
			isHit = intersectBVH(&ctxt, shadowRay, &isect);
		}
		
		isHit = AT_NAME::scene::hitLight(
			isHit, 
			light->attrib,
			lightobj,
			shadowRay.distToLight,
			distHitObjToRayOrg,
			isect.t,
			hitobj);

		if (isHit) {
			paths[idx].contrib += shadowRay.lightcontrib;
		}
	}
}

__global__ void gather(
	cudaSurfaceObject_t outSurface,
	const idaten::PathTracing::Path* __restrict__ paths,
	int width, int height)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	const auto& path = paths[idx];

	int sample = path.samples;

	float4 data;
#if 0
	surf2Dread(&data, outSurface, ix * sizeof(float4), iy);

	// First data.w value is 0.
	int n = data.w;
	data = n * data + make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 0) / sample;
	data /= (n + 1);
	data.w = n + 1;
#else
	data = make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 0) / sample;
#endif

	surf2Dwrite(
		data,
		outSurface,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}

namespace idaten {
	void PathTracing::prepare()
	{
	}

	void PathTracing::update(
		GLuint gltex,
		int width, int height,
		const aten::CameraParameter& camera,
		const std::vector<aten::ShapeParameter>& shapes,
		const std::vector<aten::MaterialParameter>& mtrls,
		const std::vector<aten::LightParameter>& lights,
		const std::vector<std::vector<aten::BVHNode>>& nodes,
		const std::vector<aten::PrimitiveParamter>& prims,
		const std::vector<aten::vertex>& vtxs,
		const std::vector<aten::mat4>& mtxs,
		const std::vector<TextureResource>& texs,
		const EnvmapResource& envmapRsc)
	{
		idaten::Renderer::update(
			gltex,
			width, height,
			camera,
			shapes,
			mtrls,
			lights,
			nodes,
			prims,
			vtxs,
			mtxs,
			texs, envmapRsc);

		m_hitbools.init(width * height);
		m_hitidx.init(width * height);
	}

	static bool doneSetStackSize = false;

	void PathTracing::render(
		aten::vec4* image,
		int width, int height,
		int maxSamples,
		int maxDepth)
	{
#ifdef __AT_DEBUG__
		if (!doneSetStackSize) {
			size_t val = 0;
			cudaThreadGetLimit(&val, cudaLimitStackSize);
			cudaThreadSetLimit(cudaLimitStackSize, val * 2);
			doneSetStackSize = true;
		}
#endif

		int depth = 0;

		paths.init(width * height);
		isects.init(width * height);
		rays.init(width * height);
		shadowRays.init(width * height);

		cudaMemset(paths.ptr(), 0, paths.bytes());

		CudaGLResourceMap rscmap(&glimg);
		auto outputSurf = glimg.bind();

		auto vtxTexPos = vtxparamsPos.bind();
		auto vtxTexNml = vtxparamsNml.bind();

		{
			std::vector<cudaTextureObject_t> tmp;
			for (int i = 0; i < nodeparam.size(); i++) {
				auto nodeTex = nodeparam[i].bind();
				tmp.push_back(nodeTex);
			}
			nodetex.writeByNum(&tmp[0], tmp.size());
		}

		if (!texRsc.empty())
		{
			std::vector<cudaTextureObject_t> tmp;
			for (int i = 0; i < texRsc.size(); i++) {
				auto cudaTex = texRsc[i].bind();
				tmp.push_back(cudaTex);
			}
			tex.writeByNum(&tmp[0], tmp.size());
		}

		static const int rrDepth = 3;

		auto time = AT_NAME::timer::getSystemTime();

		for (int i = 0; i < maxSamples; i++) {
			//int seed = time.milliSeconds;
			int seed = 0;

			onGenPath(
				width, height,
				i, maxSamples,
				seed,
				vtxTexPos);

			depth = 0;

			while (depth < maxDepth) {
				onHitTest(
					width, height,
					vtxTexPos);
				
				onShadeMiss(width, height, depth);

				int hitcount = 0;
				idaten::Compaction::compact(
					m_hitidx,
					m_hitbools,
					&hitcount);

				//AT_PRINTF("%d\n", hitcount);

				if (hitcount == 0) {
					break;
				}

				onShade(
					outputSurf,
					hitcount,
					depth, rrDepth,
					vtxTexPos, vtxTexNml);

				depth++;
			}
		}

		onGather(outputSurf, width, height, maxSamples);

		checkCudaErrors(cudaDeviceSynchronize());

		{
			vtxparamsPos.unbind();
			vtxparamsNml.unbind();

			for (int i = 0; i < nodeparam.size(); i++) {
				nodeparam[i].unbind();
			}
			nodetex.reset();

			for (int i = 0; i < texRsc.size(); i++) {
				texRsc[i].unbind();
			}
			tex.reset();
		}


		//dst.read(image, sizeof(aten::vec4) * width * height);
	}

	void PathTracing::onGenPath(
		int width, int height,
		int sample, int maxSamples,
		int seed,
		cudaTextureObject_t texVtxPos)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		genPath << <grid, block >> > (
			paths.ptr(),
			rays.ptr(),
			width, height,
			sample, maxSamples,
			seed,
			cam.ptr());

		checkCudaKernel(genPath);
	}

	void PathTracing::onHitTest(
		int width, int height,
		cudaTextureObject_t texVtxPos)
	{
		dim3 blockPerGrid_HitTest((width * height + 128 - 1) / 128);
		dim3 threadPerBlock_HitTest(128);

#ifdef ENABLE_PERSISTENT_THREAD
		hitTest << <NUM_BLOCK, dim3(WARP_SIZE, NUM_WARP_PER_BLOCK) >> > (
#else
		hitTest << <blockPerGrid_HitTest, threadPerBlock_HitTest >> > (
#endif
			//hitTest << <1, 1 >> > (
			paths.ptr(),
			isects.ptr(),
			rays.ptr(),
			m_hitbools.ptr(),
			width, height,
			shapeparam.ptr(), shapeparam.num(),
			mtrlparam.ptr(),
			lightparam.ptr(), lightparam.num(),
			nodetex.ptr(),
			primparams.ptr(),
			texVtxPos,
			mtxparams.ptr());

		checkCudaKernel(hitTest);
	}

	void PathTracing::onShadeMiss(
		int width, int height,
		int depth)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		if (m_envmapRsc.idx >= 0) {
			if (depth == 0) {
				shadeMissWithEnvmap<true> << <grid, block >> > (
					tex.ptr(),
					m_envmapRsc.idx, m_envmapRsc.avgIllum,
					paths.ptr(),
					rays.ptr(),
					width, height);
			}
			else {
				shadeMissWithEnvmap<false> << <grid, block >> > (
					tex.ptr(),
					m_envmapRsc.idx, m_envmapRsc.avgIllum,
					paths.ptr(),
					rays.ptr(),
					width, height);
			}
		}
		else {
			if (depth == 0) {
				shadeMiss<true> << <grid, block >> > (
					paths.ptr(),
					width, height);
			}
			else {
				shadeMiss<false> << <grid, block >> > (
					paths.ptr(),
					width, height);
			}
		}

		checkCudaKernel(shadeMiss);
	}

	void PathTracing::onShade(
		cudaSurfaceObject_t outputSurf,
		int hitcount,
		int depth, int rrDepth,
		cudaTextureObject_t texVtxPos,
		cudaTextureObject_t texVtxNml)
	{
		dim3 blockPerGrid((hitcount + 64 - 1) / 64);
		dim3 threadPerBlock(64);

		shade << <blockPerGrid, threadPerBlock >> > (
		//shade << <1, 1 >> > (
			outputSurf,
			paths.ptr(),
			m_hitidx.ptr(), hitcount,
			isects.ptr(),
			rays.ptr(),
			depth, rrDepth,
			shapeparam.ptr(), shapeparam.num(),
			mtrlparam.ptr(),
			lightparam.ptr(), lightparam.num(),
			nodetex.ptr(),
			primparams.ptr(),
			texVtxPos, texVtxNml,
			mtxparams.ptr(),
			tex.ptr());

		checkCudaKernel(shade);

#if 0
		hitShadowRay << <blockPerGrid, threadPerBlock >> > (
			//hitShadowRay << <1, 1 >> > (
			paths.ptr(),
			m_hitidx.ptr(), hitcount,
			shadowRays.ptr(),
			shapeparam.ptr(), shapeparam.num(),
			mtrlparam.ptr(),
			lightparam.ptr(), lightparam.num(),
			nodetex.ptr(),
			primparams.ptr(),
			vtxTexPos,
			mtxparams.ptr());

		checkCudaKernel(hitShadowRay);
#endif
	}

	void PathTracing::onGather(
		cudaSurfaceObject_t outputSurf,
		int width, int height,
		int maxSamples)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		gather << <grid, block >> > (
			outputSurf,
			paths.ptr(),
			width, height);
	}
}
