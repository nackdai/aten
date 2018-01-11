#include "svgf/svgf.h"

#include "kernel/context.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/intersect.cuh"
#include "kernel/accelerator.cuh"
#include "kernel/pt_common.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

#define ENABLE_PERSISTENT_THREAD
#define SEPARATE_SHADOWRAY_HITTEST

//#define ENABLE_DEBUG_1PIXEL

#ifdef ENABLE_DEBUG_1PIXEL
#define DEBUG_IX	(140)
#define DEBUG_IY	(511 - 81)
#endif

template <bool isFillAOV>
__global__ void genPath(
	idaten::SVGFPathTracing::Path* paths,
	aten::ray* rays,
	int width, int height,
	int sample, int maxSamples,
	unsigned int frame,
	const aten::CameraParameter* __restrict__ camera,
	const unsigned int* sobolmatrices,
	unsigned int* random)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	auto& path = paths[idx];
	path.isHit = false;

	if (path.isKill) {
		path.isTerminate = true;
		return;
	}

#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
	auto scramble = random[idx] * 0x1fe3434f;
	path.sampler.init(frame, 0, scramble, sobolmatrices);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
	auto rnd = random[idx];
	auto scramble = rnd * 0x1fe3434f * ((frame + 133 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
	path.sampler.init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 0, scramble);
#endif

	float r1 = path.sampler.nextSample();
	float r2 = path.sampler.nextSample();

	if (isFillAOV) {
		r1 = r2 = 0.5f;
	}

	float s = (ix + r1) / (float)(camera->width);
	float t = (iy + r2) / (float)(camera->height);

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

__device__ unsigned int g_headDev = 0;

__global__ void hitTest(
	idaten::SVGFPathTracing::Path* paths,
	aten::Intersection* isects,
	aten::ray* rays,
	int* hitbools,
	int width, int height,
	const aten::GeomParameter* __restrict__ shapes, int geomnum,
	const aten::LightParameter* __restrict__ lights, int lightnum,
	cudaTextureObject_t* nodes,
	const aten::PrimitiveParamter* __restrict__ prims,
	cudaTextureObject_t vtxPos,
	aten::mat4* matrices,
	int bounce,
	float hitDistLimit)
{
#ifdef ENABLE_PERSISTENT_THREAD
	// warp-wise head index of tasks in a block
	__shared__ volatile unsigned int headBlock[NUM_WARP_PER_BLOCK];

	volatile unsigned int& headWarp = headBlock[threadIdx.y];

	if (blockIdx.x == 0 && threadIdx.x == 0) {
		g_headDev = 0;
	}

	Context ctxt;
	{
		ctxt.geomnum = geomnum;
		ctxt.shapes = shapes;
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
			headWarp = atomicAdd(&g_headDev, WARP_SIZE);
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

		float t_max = AT_MATH_INF;

		if (bounce >= 1
			&& !path.isSingular)
		{
			t_max = hitDistLimit;
		}

		bool isHit = intersectClosest(&ctxt, rays[idx], &isect, t_max);

		isects[idx].t = isect.t;
		isects[idx].objid = isect.objid;
		isects[idx].mtrlid = isect.mtrlid;
		isects[idx].meshid = isect.meshid;
		isects[idx].primid = isect.primid;
		isects[idx].a = isect.a;
		isects[idx].b = isect.b;

		if (bounce >= 1
			&& !path.isSingular
			&& isect.t > hitDistLimit)
		{
			isHit = false;
		}

		path.isHit = isHit;

		hitbools[idx] = isHit ? 1 : 0;
	} while (true);
#else
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

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
		ctxt.lightnum = lightnum;
		ctxt.lights = lights;
		ctxt.nodes = nodes;
		ctxt.prims = prims;
		ctxt.vtxPos = vtxPos;
		ctxt.matrices = matrices;
	}

	aten::Intersection isect;

	float t_max = AT_MATH_INF;

	if (bounce >= 1
		&& !path.isSingular)
	{
		t_max = hitDistLimit;
	}

	bool isHit = intersectClosest(&ctxt, rays[idx], &isect, t_max);

	isects[idx].t = isect.t;
	isects[idx].objid = isect.objid;
	isects[idx].mtrlid = isect.mtrlid;
	isects[idx].meshid = isect.meshid;
	isects[idx].area = isect.area;
	isects[idx].primid = isect.primid;
	isects[idx].a = isect.a;
	isects[idx].b = isect.b;

	if (bounce >= 1
		&& !path.isSingular
		&& isect.t > hitDistLimit)
	{
		isHit = false;
	}

	path.isHit = isHit;

	hitbools[idx] = isHit ? 1 : 0;
#endif
}

template <bool isFirstBounce>
__global__ void shadeMiss(
	cudaSurfaceObject_t aovExportBuffer,
	float4* aovNormalDepth,
	float4* aovTexclrTemporalWeight,
	float4* aovMomentMeshid,
	idaten::SVGFPathTracing::Path* paths,
	int width, int height)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	auto& path = paths[idx];

	if (!path.isTerminate && !path.isHit) {
		// TODO
		auto bg = aten::vec3(0);

		if (isFirstBounce) {
			path.isKill = true;

			// Export bg color to albedo buffer.
			aovTexclrTemporalWeight[idx] = make_float4(bg.x, bg.y, bg.z, aovTexclrTemporalWeight[idx].w);
			aovNormalDepth[idx].w = -1;
			aovMomentMeshid[idx].w = -1;

			// For exporting separated albedo.
			bg = aten::vec3(1, 1, 1);

			if (aovExportBuffer > 0) {
				surf2Dwrite(
					aovNormalDepth[idx],
					aovExportBuffer,
					ix * sizeof(float4), iy,
					cudaBoundaryModeTrap);
			}
		}

		path.contrib += path.throughput * bg;

		path.isTerminate = true;
	}
}

template <bool isFirstBounce>
__global__ void shadeMissWithEnvmap(
	cudaSurfaceObject_t aovExportBuffer,
	float4* aovNormalDepth,
	float4* aovTexclrTemporalWeight,
	float4* aovMomentMeshid,
	cudaTextureObject_t* textures,
	int envmapIdx,
	real envmapAvgIllum,
	real envmapMultiplyer,
	idaten::SVGFPathTracing::Path* paths,
	const aten::ray* __restrict__ rays,
	int width, int height)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height) {
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

			// Export envmap to albedo buffer.
			aovTexclrTemporalWeight[idx] = make_float4(emit.x, emit.y, emit.z, aovTexclrTemporalWeight[idx].w);
			aovNormalDepth[idx].w = -1;
			aovMomentMeshid[idx].w = -1;

			// For exporting separated albedo.
			emit = aten::vec3(1, 1, 1);

			if (aovExportBuffer > 0) {
				surf2Dwrite(
					aovNormalDepth[idx],
					aovExportBuffer,
					ix * sizeof(float4), iy,
					cudaBoundaryModeTrap);
			}
		}
		else {
			auto pdfLight = AT_NAME::ImageBasedLight::samplePdf(emit, envmapAvgIllum);
			misW = path.pdfb / (pdfLight + path.pdfb);

			emit *= envmapMultiplyer;
		}

		path.contrib += path.throughput * misW * emit;

		path.isTerminate = true;
	}
}

template <bool isFirstBounce, int ShadowRayNum>
__global__ void shade(
	float4* aovNormalDepth,
	float4* aovTexclrTemporalWeight,
	float4* aovMomentMeshid,
	cudaSurfaceObject_t aovExportBuffer,
	aten::mat4 mtxW2C,
	int width, int height,
	idaten::SVGFPathTracing::Path* paths,
	const int* __restrict__ hitindices,
	int hitnum,
	const aten::Intersection* __restrict__ isects,
	aten::ray* rays,
	int frame,
	int bounce, int rrBounce,
	const aten::GeomParameter* __restrict__ shapes, int geomnum,
	const aten::MaterialParameter* __restrict__ mtrls,
	const aten::LightParameter* __restrict__ lights, int lightnum,
	cudaTextureObject_t* nodes,
	const aten::PrimitiveParamter* __restrict__ prims,
	cudaTextureObject_t vtxPos,
	cudaTextureObject_t vtxNml,
	const aten::mat4* __restrict__ matrices,
	cudaTextureObject_t* textures,
	unsigned int* random,
	idaten::SVGFPathTracing::ShadowRay* shadowRays)
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

#ifdef ENABLE_DEBUG_1PIXEL
	int ix = DEBUG_IX;
	int iy = DEBUG_IY;
	idx = getIdx(ix, iy, width);
#endif

	__shared__ idaten::SVGFPathTracing::Path shPaths[64];
	__shared__ idaten::SVGFPathTracing::ShadowRay shShadowRays[64];
	__shared__ aten::MaterialParameter shMtrls[64];

	shPaths[threadIdx.x] = paths[idx];

	const auto ray = rays[idx];

#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
	auto scramble = random[idx] * 0x1fe3434f;
	shPaths[threadIdx.x].sampler.init(frame, 4 + bounce * 300, scramble);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
	auto rnd = random[idx];
	auto scramble = rnd * 0x1fe3434f * ((frame + 331 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
	shPaths[threadIdx.x].sampler.init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 4 + bounce * 300, scramble);
#endif

	aten::hitrecord rec;

	const auto& isect = isects[idx];

	auto obj = &ctxt.shapes[isect.objid];
	evalHitResult(&ctxt, obj, ray, &rec, &isect);

	shMtrls[threadIdx.x] = ctxt.mtrls[rec.mtrlid];

	bool isBackfacing = dot(rec.normal, -ray.dir) < 0.0f;

	// 交差位置の法線.
	// 物体からのレイの入出を考慮.
	aten::vec3 orienting_normal = rec.normal;

	if (shMtrls[threadIdx.x].type != aten::MaterialType::Layer) {
		shMtrls[threadIdx.x].albedoMap = (int)(shMtrls[threadIdx.x].albedoMap >= 0 ? ctxt.textures[shMtrls[threadIdx.x].albedoMap] : -1);
		shMtrls[threadIdx.x].normalMap = (int)(shMtrls[threadIdx.x].normalMap >= 0 ? ctxt.textures[shMtrls[threadIdx.x].normalMap] : -1);
		shMtrls[threadIdx.x].roughnessMap = (int)(shMtrls[threadIdx.x].roughnessMap >= 0 ? ctxt.textures[shMtrls[threadIdx.x].roughnessMap] : -1);
	}

	// Render AOVs.
	// NOTE
	// 厳密に法線をAOVに保持するなら、法線マップ適用後するべき.
	// しかし、temporal reprojection、atrousなどのフィルタ適用時に法線を参照する際に、法線マップが細かすぎてはじかれてしまうことがある.
	// それにより、フィルタがおもったようにかからずフィルタの品質が下がってしまう問題が発生する.
	if (isFirstBounce) {
		int ix = idx % width;
		int iy = idx / width;

		// World coordinate to Clip coordinate.
		aten::vec4 pos = aten::vec4(rec.p, 1);
		pos = mtxW2C.apply(pos);

		// normal, depth
		aovNormalDepth[idx] = make_float4(orienting_normal.x, orienting_normal.y, orienting_normal.z, pos.w);

		// meshid.
		aovMomentMeshid[idx].w = isect.meshid;

		// texture color.
		auto texcolor = AT_NAME::material::sampleTexture(shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, 1.0f);
		aovTexclrTemporalWeight[idx] = make_float4(texcolor.x, texcolor.y, texcolor.z, aovTexclrTemporalWeight[idx].w);

		// For exporting separated albedo.
		shMtrls[threadIdx.x].albedoMap = -1;

		if (aovExportBuffer > 0) {
			surf2Dwrite(
				aovNormalDepth[idx],
				aovExportBuffer,
				ix * sizeof(float4), iy,
				cudaBoundaryModeTrap);
		}
	}

	// Implicit conection to light.
	if (shMtrls[threadIdx.x].attrib.isEmissive) {
		if (!isBackfacing) {
			float weight = 1.0f;

			if (bounce > 0 && !shPaths[threadIdx.x].isSingular) {
				auto cosLight = dot(orienting_normal, -ray.dir);
				auto dist2 = aten::squared_length(rec.p - ray.org);

				if (cosLight >= 0) {
					auto pdfLight = 1 / rec.area;

					// Convert pdf area to sradian.
					// http://www.slideshare.net/h013/edubpt-v100
					// p31 - p35
					pdfLight = pdfLight * dist2 / cosLight;

					weight = shPaths[threadIdx.x].pdfb / (pdfLight + shPaths[threadIdx.x].pdfb);
				}
			}

			shPaths[threadIdx.x].contrib += shPaths[threadIdx.x].throughput * weight * shMtrls[threadIdx.x].baseColor;
		}

		// When ray hit the light, tracing will finish.
		shPaths[threadIdx.x].isTerminate = true;
		paths[idx] = shPaths[threadIdx.x];
		return;
	}

	if (!shMtrls[threadIdx.x].attrib.isTranslucent && isBackfacing) {
		orienting_normal = -orienting_normal;
	}

	// Apply normal map.
	int normalMap = shMtrls[threadIdx.x].normalMap;
	if (shMtrls[threadIdx.x].type == aten::MaterialType::Layer) {
		// 最表層の NormalMap を適用.
		auto* topmtrl = &ctxt.mtrls[shMtrls[threadIdx.x].layer[0]];
		normalMap = (int)(topmtrl->normalMap >= 0 ? ctxt.textures[topmtrl->normalMap] : -1);
	}
	AT_NAME::material::applyNormalMap(normalMap, orienting_normal, orienting_normal, rec.u, rec.v);

#ifdef SEPARATE_SHADOWRAY_HITTEST
	shShadowRays[threadIdx.x].isActive = false;
#endif

	auto albedo = AT_NAME::sampleTexture(shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, aten::vec3(1));

#if 1
	// Explicit conection to light.
	if (!shMtrls[threadIdx.x].attrib.isSingular)
	{
		for (int i = 0; i < ShadowRayNum; i++) {
			real lightSelectPdf = 1;
			aten::LightSampleResult sampleres;

			// TODO
			// Importance sampling.
			int lightidx = aten::cmpMin<int>(shPaths[threadIdx.x].sampler.nextSample() * lightnum, lightnum - 1);
			lightSelectPdf = 1.0f / lightnum;

			aten::LightParameter light;
			light.pos = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 0];
			light.dir = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 1];
			light.le = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 2];
			light.v0 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 3];
			light.v1 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 4];
			light.v2 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 5];
			//auto light = ctxt.lights[lightidx];

			sampleLight(&sampleres, &ctxt, &light, rec.p, orienting_normal, &shPaths[threadIdx.x].sampler);

			const auto& posLight = sampleres.pos;
			const auto& nmlLight = sampleres.nml;
			real pdfLight = sampleres.pdf;

			auto dirToLight = normalize(sampleres.dir);
			auto distToLight = length(posLight - rec.p);

			auto shadowRayOrg = rec.p + AT_MATH_EPSILON * orienting_normal;
			auto tmp = rec.p + dirToLight - shadowRayOrg;
			auto shadowRayDir = normalize(tmp);

#ifdef SEPARATE_SHADOWRAY_HITTEST
			shShadowRays[threadIdx.x].isActive = true;
			shShadowRays[threadIdx.x].ray[i] = aten::ray(shadowRayOrg, shadowRayDir);
			shShadowRays[threadIdx.x].targetLightId[i] = lightidx;
			shShadowRays[threadIdx.x].distToLight[i] = distToLight;
			shShadowRays[threadIdx.x].lightcontrib[i] = aten::vec3(0);
#else
			auto lightobj = sampleres.obj;

			real distHitObjToRayOrg = AT_MATH_INF;

			// Ray aim to the area light.
			// So, if ray doesn't hit anything in intersectCloserBVH, ray hit the area light.
			auto hitobj = lightobj;

			aten::Intersection isectTmp;

			aten::ray shadowRay(shadowRayOrg, shadowRayDir);

			bool isHit = intersectCloser(&ctxt, shadowRay, &isectTmp, distToLight - AT_MATH_EPSILON);

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

			if (isHit)
#endif
			{
				auto cosShadow = dot(orienting_normal, dirToLight);

				real pdfb = samplePDF(&ctxt, &shMtrls[threadIdx.x], orienting_normal, ray.dir, dirToLight, rec.u, rec.v);
				auto bsdf = sampleBSDF(&ctxt, &shMtrls[threadIdx.x], orienting_normal, ray.dir, dirToLight, rec.u, rec.v, albedo);

				bsdf *= shPaths[threadIdx.x].throughput;

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
#ifdef SEPARATE_SHADOWRAY_HITTEST
						shShadowRays[threadIdx.x].lightcontrib[i] =
#else
						shPaths[threadIdx.x].contrib +=
#endif
							(misW * bsdf * emit * cosShadow / pdfLight) / lightSelectPdf / (float)ShadowRayNum;
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
#ifdef SEPARATE_SHADOWRAY_HITTEST
							shShadowRays[threadIdx.x].lightcontrib[i] =
#else
							shPaths[threadIdx.x].contrib +=
#endif
								(misW * (bsdf * emit * G) / pdfLight) / lightSelectPdf / (float)ShadowRayNum;
						}
					}
				}
			}
		}
	}
#endif

	real russianProb = real(1);

	if (bounce > rrBounce) {
		auto t = normalize(shPaths[threadIdx.x].throughput);
		auto p = aten::cmpMax(t.r, aten::cmpMax(t.g, t.b));

		russianProb = shPaths[threadIdx.x].sampler.nextSample();

		if (russianProb >= p) {
			//shPaths[threadIdx.x].contrib = aten::vec3(0);
			shPaths[threadIdx.x].isTerminate = true;
		}
		else {
			russianProb = p;
		}
	}
			
	AT_NAME::MaterialSampling sampling;

	sampleMaterial(
		&sampling,
		&ctxt,
		&shMtrls[threadIdx.x],
		orienting_normal,
		ray.dir,
		rec.normal,
		&shPaths[threadIdx.x].sampler,
		rec.u, rec.v,
		albedo);

	auto nextDir = normalize(sampling.dir);
	auto pdfb = sampling.pdf;
	auto bsdf = sampling.bsdf;

	real c = 1;
	if (!shMtrls[threadIdx.x].attrib.isSingular) {
		// TODO
		// AMDのはabsしているが....
		//c = aten::abs(dot(orienting_normal, nextDir));
		c = dot(orienting_normal, nextDir);
	}

	if (pdfb > 0 && c > 0) {
		shPaths[threadIdx.x].throughput *= bsdf * c / pdfb;
		shPaths[threadIdx.x].throughput /= russianProb;
	}
	else {
		shPaths[threadIdx.x].isTerminate = true;
	}

	// Make next ray.
	rays[idx] = aten::ray(rec.p, nextDir);

	shPaths[threadIdx.x].pdfb = pdfb;
	shPaths[threadIdx.x].isSingular = shMtrls[threadIdx.x].attrib.isSingular;

	paths[idx] = shPaths[threadIdx.x];
	shadowRays[idx] = shShadowRays[threadIdx.x];
}

template <int ShadowRayNum>
__global__ void hitShadowRay(
	idaten::SVGFPathTracing::Path* paths,
	int* hitindices,
	int hitnum,
	const idaten::SVGFPathTracing::ShadowRay* __restrict__ shadowRays,
	const aten::GeomParameter* __restrict__ shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	const aten::LightParameter* __restrict__ lights, int lightnum,
	cudaTextureObject_t* nodes,
	const aten::PrimitiveParamter* __restrict__ prims,
	cudaTextureObject_t vtxPos,
	const aten::mat4* __restrict__ matrices)
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
#pragma unroll
		for (int i = 0; i < ShadowRayNum; i++) {
			auto targetLightId = shadowRay.targetLightId[i];
			auto distToLight = shadowRay.distToLight[i];

			auto light = &ctxt.lights[targetLightId];
			auto lightobj = (light->objid >= 0 ? &ctxt.shapes[light->objid] : nullptr);

			real distHitObjToRayOrg = AT_MATH_INF;

			// Ray aim to the area light.
			// So, if ray doesn't hit anything in intersectCloserBVH, ray hit the area light.
			const aten::GeomParameter* hitobj = lightobj;

			aten::Intersection isectTmp;

			bool isHit = false;
			isHit = intersectCloser(&ctxt, shadowRay.ray[i], &isectTmp, distToLight - AT_MATH_EPSILON);

			if (isHit) {
				hitobj = &ctxt.shapes[isectTmp.objid];
			}

			isHit = AT_NAME::scene::hitLight(
				isHit,
				light->attrib,
				lightobj,
				distToLight,
				distHitObjToRayOrg,
				isectTmp.t,
				hitobj);

			if (isHit) {
				paths[idx].contrib += shadowRay.lightcontrib[i];
			}
		}
	}
}

__global__ void gather(
	cudaSurfaceObject_t dst,
	float4* aovColorVariance,
	float4* aovMomentMeshid,
	const idaten::SVGFPathTracing::Path* __restrict__ paths,
	int width, int height)
{
	auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height) {
		return;
	}

#ifdef ENABLE_DEBUG_1PIXEL
	ix = DEBUG_IX;
	iy = DEBUG_IY;
#endif

	const auto idx = getIdx(ix, iy, width);

	const auto& path = paths[idx];

	int sample = path.samples;

	float3 contrib = make_float3(path.contrib.x, path.contrib.y, path.contrib.z) / sample;
	//contrib.w = sample;

	float lum = AT_NAME::color::luminance(contrib.x, contrib.y, contrib.z);

	aovMomentMeshid[idx].x += lum * lum;
	aovMomentMeshid[idx].y += lum;
	aovMomentMeshid[idx].z += 1;

	aovColorVariance[idx] = make_float4(contrib.x, contrib.y, contrib.z, aovColorVariance[idx].w);

#if 0
	auto n = aovs[idx].moments.w;

	auto m = aovs[idx].moments / n;

	auto var = m.x - m.y * m.y;

	surf2Dwrite(
		make_float4(var, var, var, 1),
		dst,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
#else
	surf2Dwrite(
		make_float4(contrib, 0),
		dst,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
#endif
}

namespace idaten
{
	void SVGFPathTracing::onGenPath(
		int width, int height,
		int sample, int maxSamples,
		int seed,
		cudaTextureObject_t texVtxPos,
		cudaTextureObject_t texVtxNml)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		if (m_mode == Mode::AOVar) {
			genPath<true> << <grid, block >> > (
				m_paths.ptr(),
				m_rays.ptr(),
				width, height,
				sample, maxSamples,
				m_frame,
				m_cam.ptr(),
				m_sobolMatrices.ptr(),
				m_random.ptr());
		}
		else {
			genPath<false> << <grid, block >> > (
				m_paths.ptr(),
				m_rays.ptr(),
				width, height,
				sample, maxSamples,
				m_frame,
				m_cam.ptr(),
				m_sobolMatrices.ptr(),
				m_random.ptr());
		}

		checkCudaKernel(genPath);
	}

	void SVGFPathTracing::onHitTest(
		int width, int height,
		int bounce,
		cudaTextureObject_t texVtxPos)
	{
		if (bounce == 0) {
			onScreenSpaceHitTest(width, height, bounce, texVtxPos);
		}
		else {
#ifdef ENABLE_PERSISTENT_THREAD
			hitTest << <NUM_BLOCK, dim3(WARP_SIZE, NUM_WARP_PER_BLOCK) >> > (
#else
			dim3 block(BLOCK_SIZE, BLOCK_SIZE);
			dim3 grid(
				(width + block.x - 1) / block.x,
				(height + block.y - 1) / block.y);

			hitTest << <grid, block >> > (
#endif
				//hitTest << <1, 1 >> > (
				m_paths.ptr(),
				m_isects.ptr(),
				m_rays.ptr(),
				m_hitbools.ptr(),
				width, height,
				m_shapeparam.ptr(), m_shapeparam.num(),
				m_lightparam.ptr(), m_lightparam.num(),
				m_nodetex.ptr(),
				m_primparams.ptr(),
				texVtxPos,
				m_mtxparams.ptr(),
				bounce,
				m_hitDistLimit);

			checkCudaKernel(hitTest);
		}
	}

	void SVGFPathTracing::onShadeMiss(
		int width, int height,
		int bounce,
		cudaSurfaceObject_t aovExportBuffer)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		int curaov = getCurAovs();

		if (m_envmapRsc.idx >= 0) {
			if (bounce == 0) {
				shadeMissWithEnvmap<true> << <grid, block >> > (
					aovExportBuffer,
					m_aovNormalDepth[curaov].ptr(),
					m_aovTexclrTemporalWeight[curaov].ptr(),
					m_aovMomentMeshid[curaov].ptr(),
					m_tex.ptr(),
					m_envmapRsc.idx, m_envmapRsc.avgIllum, m_envmapRsc.multiplyer,
					m_paths.ptr(),
					m_rays.ptr(),
					width, height);
			}
			else {
				shadeMissWithEnvmap<false> << <grid, block >> > (
					aovExportBuffer,
					m_aovNormalDepth[curaov].ptr(),
					m_aovTexclrTemporalWeight[curaov].ptr(),
					m_aovMomentMeshid[curaov].ptr(),
					m_tex.ptr(),
					m_envmapRsc.idx, m_envmapRsc.avgIllum, m_envmapRsc.multiplyer,
					m_paths.ptr(),
					m_rays.ptr(),
					width, height);
			}
		}
		else {
			if (bounce == 0) {
				shadeMiss<true> << <grid, block >> > (
					aovExportBuffer,
					m_aovNormalDepth[curaov].ptr(),
					m_aovTexclrTemporalWeight[curaov].ptr(),
					m_aovMomentMeshid[curaov].ptr(),
					m_paths.ptr(),
					width, height);
			}
			else {
				shadeMiss<false> << <grid, block >> > (
					aovExportBuffer,
					m_aovNormalDepth[curaov].ptr(),
					m_aovTexclrTemporalWeight[curaov].ptr(),
					m_aovMomentMeshid[curaov].ptr(),
					m_paths.ptr(),
					width, height);
			}
		}

		checkCudaKernel(shadeMiss);
	}

	void SVGFPathTracing::onShade(
		cudaSurfaceObject_t outputSurf,
		cudaSurfaceObject_t aovExportBuffer,
		int hitcount,
		int width, int height,
		int bounce, int rrBounce,
		cudaTextureObject_t texVtxPos,
		cudaTextureObject_t texVtxNml)
	{
		m_mtxW2V.lookat(
			m_camParam.origin,
			m_camParam.center,
			m_camParam.up);

		m_mtxV2C.perspective(
			m_camParam.znear,
			m_camParam.zfar,
			m_camParam.vfov,
			m_camParam.aspect);

		m_mtxC2V = m_mtxV2C;
		m_mtxC2V.invert();

		m_mtxV2W = m_mtxW2V;
		m_mtxV2W.invert();

		aten::mat4 mtxW2C = m_mtxV2C * m_mtxW2V;

#ifdef ENABLE_DEBUG_1PIXEL
		int blockPerGrid = 1;
		int threadPerBlock = 1;
#else
		dim3 blockPerGrid((hitcount + 64 - 1) / 64);
		dim3 threadPerBlock(64);
#endif

		int curaov = getCurAovs();

		if (bounce == 0) {
			shade<true, ShdowRayNum> << <blockPerGrid, threadPerBlock >> > (
				m_aovNormalDepth[curaov].ptr(),
				m_aovTexclrTemporalWeight[curaov].ptr(),
				m_aovMomentMeshid[curaov].ptr(),
				aovExportBuffer,
				mtxW2C,
				width, height,
				m_paths.ptr(),
				m_hitidx.ptr(), hitcount,
				m_isects.ptr(),
				m_rays.ptr(),
				m_frame,
				bounce, rrBounce,
				m_shapeparam.ptr(), m_shapeparam.num(),
				m_mtrlparam.ptr(),
				m_lightparam.ptr(), m_lightparam.num(),
				m_nodetex.ptr(),
				m_primparams.ptr(),
				texVtxPos, texVtxNml,
				m_mtxparams.ptr(),
				m_tex.ptr(),
				m_random.ptr(),
				m_shadowRays.ptr());
		}
		else {
			shade<false, ShdowRayNum> << <blockPerGrid, threadPerBlock >> > (
				m_aovNormalDepth[curaov].ptr(),
				m_aovTexclrTemporalWeight[curaov].ptr(),
				m_aovMomentMeshid[curaov].ptr(),
				aovExportBuffer,
				mtxW2C,
				width, height,
				m_paths.ptr(),
				m_hitidx.ptr(), hitcount,
				m_isects.ptr(),
				m_rays.ptr(),
				m_frame,
				bounce, rrBounce,
				m_shapeparam.ptr(), m_shapeparam.num(),
				m_mtrlparam.ptr(),
				m_lightparam.ptr(), m_lightparam.num(),
				m_nodetex.ptr(),
				m_primparams.ptr(),
				texVtxPos, texVtxNml,
				m_mtxparams.ptr(),
				m_tex.ptr(),
				m_random.ptr(),
				m_shadowRays.ptr());
		}

		checkCudaKernel(shade);

#ifdef SEPARATE_SHADOWRAY_HITTEST
		hitShadowRay<ShdowRayNum> << <blockPerGrid, threadPerBlock >> > (
			m_paths.ptr(),
			m_hitidx.ptr(), hitcount,
			m_shadowRays.ptr(),
			m_shapeparam.ptr(), m_shapeparam.num(),
			m_mtrlparam.ptr(),
			m_lightparam.ptr(), m_lightparam.num(),
			m_nodetex.ptr(),
			m_primparams.ptr(),
			texVtxPos,
			m_mtxparams.ptr());

		checkCudaKernel(hitShadowRay);
#endif
	}

	void SVGFPathTracing::onGather(
		cudaSurfaceObject_t outputSurf,
		int width, int height,
		int maxSamples)
	{
#ifdef ENABLE_DEBUG_1PIXEL
		int block = 1;
		int grid = 1;
#else
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);
#endif
		int curaov = getCurAovs();

		if (m_mode == Mode::PT) {
			gather << <grid, block >> > (
				outputSurf,
				m_aovColorVariance[curaov].ptr(),
				m_aovMomentMeshid[curaov].ptr(),
				m_paths.ptr(),
				width, height);

			checkCudaKernel(gather);
		}
		else if (m_mode == Mode::AOVar) {
			onFillAOV(outputSurf, width, height);
		}
		else {
			if (isFirstFrame()) {
				gather << <grid, block >> > (
					outputSurf,
					m_aovColorVariance[curaov].ptr(),
					m_aovMomentMeshid[curaov].ptr(),
					m_paths.ptr(),
					width, height);

				checkCudaKernel(gather);
			}
			else {
				onTemporalReprojection(
					outputSurf,
					width, height);
			}
		}

		m_mtxPrevW2V = m_mtxW2V;
	}
}
