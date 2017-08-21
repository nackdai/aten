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

//#define ENALBLE_SEPARATE_ALBEDO

__global__ void genPath(
	idaten::PathTracing::Path* paths,
	aten::ray* rays,
	int width, int height,
	int sample, int maxSamples,
	int seed,
	const aten::CameraParameter* __restrict__ camera,
	const unsigned int* sobolmatrices)
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

	path.sampler.init((iy * height * 4 + ix * 4) * maxSamples + sample + 1 + seed, sobolmatrices);

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
		isects[idx].meshid = isect.meshid;
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
	isects[idx].meshid = isect.meshid;
	isects[idx].area = isect.area;
	isects[idx].primid = isect.primid;
	isects[idx].a = isect.a;
	isects[idx].b = isect.b;

	path.isHit = isHit;

	hitbools[idx] = isHit ? 1 : 0;
#endif
}

template <bool isFirstBounce, bool needAOV>
__global__ void shadeMiss(
	cudaSurfaceObject_t* aovs,
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

		if (isFirstBounce) {
			path.isKill = true;

			if (needAOV) {
				surf2Dwrite(
					make_float4(bg.x, bg.y, bg.z, 1),
					aovs[2],
					ix * sizeof(float4), iy,
					cudaBoundaryModeTrap);
			}

#ifdef ENALBLE_SEPARATE_ALBEDO
			// For exporting separated albedo.
			bg = aten::vec3(1, 1, 1);
#endif
		}

		path.contrib += path.throughput * bg;

		path.isTerminate = true;
	}
}

template <bool isFirstBounce, bool needAOV>
__global__ void shadeMissWithEnvmap(
	cudaSurfaceObject_t* aovs,
	cudaTextureObject_t* textures,
	int envmapIdx,
	real envmapAvgIllum,
	real envmapMultiplyer,
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

			if (needAOV) {
				surf2Dwrite(
					make_float4(emit.x, emit.y, emit.z, 1),
					aovs[2],
					ix * sizeof(float4), iy,
					cudaBoundaryModeTrap);
			}

#ifdef ENALBLE_SEPARATE_ALBEDO
			// For exporting separated albedo.
			emit = aten::vec3(1, 1, 1);
#endif
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

template <bool needAOV>
__global__ void shade(
	cudaSurfaceObject_t outSurface,
	cudaSurfaceObject_t* aovs,
	float3 posRange,
	int width, int height,
	idaten::PathTracing::Path* paths,
	int* hitindices,
	int hitnum,
	const aten::Intersection* __restrict__ isects,
	aten::ray* rays,
	int bounce, int rrBounce,
	const aten::ShapeParameter* __restrict__ shapes, int geomnum,
	aten::MaterialParameter* mtrls,
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

	auto& path = paths[idx];
	const auto& ray = rays[idx];

	aten::hitrecord rec;

	const auto& isect = isects[idx];

	auto obj = &ctxt.shapes[isect.objid];
	evalHitResult(&ctxt, obj, ray, &rec, &isect);

	aten::MaterialParameter mtrl = ctxt.mtrls[rec.mtrlid];

	// 交差位置の法線.
	// 物体からのレイの入出を考慮.
	aten::vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

	// Apply normal map.
	if (mtrl.type == aten::MaterialType::Layer) {
		// 最表層の NormalMap を適用.
		auto* topmtrl = &ctxt.mtrls[mtrl.layer[0]];
		auto normalMap = (int)(topmtrl->normalMap >= 0 ? ctxt.textures[topmtrl->normalMap] : -1);
		AT_NAME::material::applyNormalMap(normalMap, orienting_normal, orienting_normal, rec.u, rec.v);
	}
	else {
		mtrl.albedoMap = (int)(mtrl.albedoMap >= 0 ? ctxt.textures[mtrl.albedoMap] : -1);
		mtrl.normalMap = (int)(mtrl.normalMap >= 0 ? ctxt.textures[mtrl.normalMap] : -1);
		mtrl.roughnessMap = (int)(mtrl.roughnessMap >= 0 ? ctxt.textures[mtrl.roughnessMap] : -1);

		AT_NAME::material::applyNormalMap(mtrl.normalMap, orienting_normal, orienting_normal, rec.u, rec.v);
	}

#if 1
	if (needAOV) {
		int ix = idx % width;
		int iy = idx / width;

		auto p = make_float3(rec.p.x, rec.p.y, rec.p.z);
		p /= posRange;

		auto n = (orienting_normal + 1.0f) * 0.5f;

		// position
		surf2Dwrite(
			make_float4(p.x, p.y, p.z, rec.mtrlid),
			aovs[0],
			ix * sizeof(float4), iy,
			cudaBoundaryModeTrap);

		// normal
		surf2Dwrite(
			make_float4(n.x, n.y, n.z, isect.meshid),
			aovs[1],
			ix * sizeof(float4), iy,
			cudaBoundaryModeTrap);

		auto albedo = AT_NAME::material::sampleTexture(mtrl.albedoMap, rec.u, rec.v, 1.0f);

		surf2Dwrite(
			make_float4(albedo.x, albedo.y, albedo.z, 1),
			aovs[2],
			ix * sizeof(float4), iy,
			cudaBoundaryModeTrap);
	}
#endif

#ifdef ENALBLE_SEPARATE_ALBEDO
	// For exporting separated albedo.
	mtrl.albedoMap = -1;
#endif

	// Implicit conection to light.
	if (mtrl.attrib.isEmissive) {
		float weight = 1.0f;

		if (bounce > 0 && !path.isSingular) {
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

		path.contrib += path.throughput * weight * mtrl.baseColor;

		// When ray hit the light, tracing will finish.
		path.isTerminate = true;
		return;
	}

	// Explicit conection to light.
	if (!mtrl.attrib.isSingular)
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

			real pdfb = samplePDF(&ctxt, &mtrl, orienting_normal, ray.dir, dirToLight, rec.u, rec.v);
			auto bsdf = sampleBSDF(&ctxt, &mtrl, orienting_normal, ray.dir, dirToLight, rec.u, rec.v);

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

	if (bounce > rrBounce) {
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
		&ctxt,
		&mtrl,
		orienting_normal,
		ray.dir,
		rec.normal,
		&path.sampler,
		rec.u, rec.v);

	auto nextDir = normalize(sampling.dir);
	auto pdfb = sampling.pdf;
	auto bsdf = sampling.bsdf;

	real c = 1;
	if (!mtrl.attrib.isSingular) {
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
	path.isSingular = mtrl.attrib.isSingular;
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
#if 1
	surf2Dread(&data, outSurface, ix * sizeof(float4), iy);

	// First data.w value is 0.
	int n = data.w;
	data = n * data + make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 0) / sample;
	data /= (n + 1);
	data.w = n + 1;
#else
	data = make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 0) / sample;
	data.w = sample;
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

		m_sobolMatrices.init(AT_COUNTOF(sobol::Matrices::matrices));
		m_sobolMatrices.writeByNum(sobol::Matrices::matrices, m_sobolMatrices.maxNum());
	}

	void PathTracing::enableRenderAOV(
		GLuint gltexPosition,
		GLuint gltexNormal,
		GLuint gltexAlbedo,
		aten::vec3& posRange)
	{
		AT_ASSERT(gltexPosition > 0);
		AT_ASSERT(gltexNormal > 0);

		if (!m_enableAOV) {
			m_enableAOV = true;

			m_posRange = posRange;

			m_aovs.resize(3);
			m_aovs[0].init(gltexPosition, CudaGLRscRegisterType::WriteOnly);
			m_aovs[1].init(gltexNormal, CudaGLRscRegisterType::WriteOnly);
			m_aovs[2].init(gltexAlbedo, CudaGLRscRegisterType::WriteOnly);

			m_aovCudaRsc.init(3);
		}
	}

	static bool doneSetStackSize = false;

	void PathTracing::render(
		aten::vec4* image,
		int width, int height,
		int maxSamples,
		int maxBounce)
	{
#ifdef __AT_DEBUG__
		if (!doneSetStackSize) {
			size_t val = 0;
			cudaThreadGetLimit(&val, cudaLimitStackSize);
			cudaThreadSetLimit(cudaLimitStackSize, val * 4);
			doneSetStackSize = true;
		}
#endif

		int bounce = 0;

		m_paths.init(width * height);
		m_isects.init(width * height);
		m_rays.init(width * height);
		m_shadowRays.init(width * height);

		cudaMemset(m_paths.ptr(), 0, m_paths.bytes());

		CudaGLResourceMap rscmap(&m_glimg);
		auto outputSurf = m_glimg.bind();

		auto vtxTexPos = m_vtxparamsPos.bind();
		auto vtxTexNml = m_vtxparamsNml.bind();

		{
			std::vector<cudaTextureObject_t> tmp;
			for (int i = 0; i < m_nodeparam.size(); i++) {
				auto nodeTex = m_nodeparam[i].bind();
				tmp.push_back(nodeTex);
			}
			m_nodetex.writeByNum(&tmp[0], tmp.size());
		}

		if (!m_texRsc.empty())
		{
			std::vector<cudaTextureObject_t> tmp;
			for (int i = 0; i < m_texRsc.size(); i++) {
				auto cudaTex = m_texRsc[i].bind();
				tmp.push_back(cudaTex);
			}
			m_tex.writeByNum(&tmp[0], tmp.size());
		}

		if (m_enableAOV) {
			std::vector<cudaSurfaceObject_t> tmp;
			for (int i = 0; i < m_aovs.size(); i++) {
				m_aovs[i].map();
				tmp.push_back(m_aovs[i].bind());
			}
			m_aovCudaRsc.writeByNum(&tmp[0], tmp.size());
		}

		static const int rrBounce = 3;

		auto time = AT_NAME::timer::getSystemTime();

		for (int i = 0; i < maxSamples; i++) {
			int seed = time.milliSeconds;
			//int seed = 0;

			onGenPath(
				width, height,
				i, maxSamples,
				seed,
				vtxTexPos,
				vtxTexNml);

			bounce = 0;

			while (bounce < maxBounce) {
				onHitTest(
					width, height,
					vtxTexPos);
				
				onShadeMiss(width, height, bounce);

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
					width, height,
					bounce, rrBounce,
					vtxTexPos, vtxTexNml);

				bounce++;
			}
		}

		onGather(outputSurf, width, height, maxSamples);

		checkCudaErrors(cudaDeviceSynchronize());

		{
			m_vtxparamsPos.unbind();
			m_vtxparamsNml.unbind();

			for (int i = 0; i < m_nodeparam.size(); i++) {
				m_nodeparam[i].unbind();
			}
			m_nodetex.reset();

			for (int i = 0; i < m_texRsc.size(); i++) {
				m_texRsc[i].unbind();
			}
			m_tex.reset();

			for (int i = 0; i < m_aovs.size(); i++) {
				m_aovs[i].unbind();
				m_aovs[i].unmap();
			}
			m_aovCudaRsc.reset();
		}
	}

	void PathTracing::onGenPath(
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

		genPath << <grid, block >> > (
			m_paths.ptr(),
			m_rays.ptr(),
			width, height,
			sample, maxSamples,
			seed,
			m_cam.ptr(),
			m_sobolMatrices.ptr());

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
			m_mtxparams.ptr());

		checkCudaKernel(hitTest);
	}

	void PathTracing::onShadeMiss(
		int width, int height,
		int bounce)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		bool enableAOV = (bounce == 0 && m_enableAOV);

		if (m_envmapRsc.idx >= 0) {
			if (bounce == 0) {
				if (enableAOV) {
					shadeMissWithEnvmap<true, true> << <grid, block >> > (
						m_aovCudaRsc.ptr(),
						m_tex.ptr(),
						m_envmapRsc.idx, m_envmapRsc.avgIllum, m_envmapRsc.multiplyer,
						m_paths.ptr(),
						m_rays.ptr(),
						width, height);
				}
				else {
					shadeMissWithEnvmap<true, false> << <grid, block >> > (
						m_aovCudaRsc.ptr(),
						m_tex.ptr(),
						m_envmapRsc.idx, m_envmapRsc.avgIllum, m_envmapRsc.multiplyer,
						m_paths.ptr(),
						m_rays.ptr(),
						width, height);
				}
			}
			else {
				shadeMissWithEnvmap<false, false> << <grid, block >> > (
					m_aovCudaRsc.ptr(),
					m_tex.ptr(),
					m_envmapRsc.idx, m_envmapRsc.avgIllum, m_envmapRsc.multiplyer,
					m_paths.ptr(),
					m_rays.ptr(),
					width, height);
			}
		}
		else {
			if (bounce == 0) {
				if (enableAOV) {
					shadeMiss<true, true> << <grid, block >> > (
						m_aovCudaRsc.ptr(),
						m_paths.ptr(),
						width, height);
				}
				else {
					shadeMiss<true, false> << <grid, block >> > (
						m_aovCudaRsc.ptr(),
						m_paths.ptr(),
						width, height);
				}
			}
			else {
				shadeMiss<false, false> << <grid, block >> > (
					m_aovCudaRsc.ptr(),
					m_paths.ptr(),
					width, height);
			}
		}

		checkCudaKernel(shadeMiss);
	}

	void PathTracing::onShade(
		cudaSurfaceObject_t outputSurf,
		int hitcount,
		int width, int height,
		int bounce, int rrBounce,
		cudaTextureObject_t texVtxPos,
		cudaTextureObject_t texVtxNml)
	{
		dim3 blockPerGrid((hitcount + 64 - 1) / 64);
		dim3 threadPerBlock(64);

		bool enableAOV = (bounce == 0 && m_enableAOV);
		float3 posRange = make_float3(m_posRange.x, m_posRange.y, m_posRange.z);

		if (enableAOV) {
			shade<true> << <blockPerGrid, threadPerBlock >> > (
			//shade<true> << <1, 1 >> > (
				outputSurf,
				m_aovCudaRsc.ptr(), posRange,
				width, height,
				m_paths.ptr(),
				m_hitidx.ptr(), hitcount,
				m_isects.ptr(),
				m_rays.ptr(),
				bounce, rrBounce,
				m_shapeparam.ptr(), m_shapeparam.num(),
				m_mtrlparam.ptr(),
				m_lightparam.ptr(), m_lightparam.num(),
				m_nodetex.ptr(),
				m_primparams.ptr(),
				texVtxPos, texVtxNml,
				m_mtxparams.ptr(),
				m_tex.ptr());
		}
		else {
			shade<false> << <blockPerGrid, threadPerBlock >> > (
			//shade<false> << <1, 1 >> > (
				outputSurf,
				m_aovCudaRsc.ptr(), posRange,
				width, height,
				m_paths.ptr(),
				m_hitidx.ptr(), hitcount,
				m_isects.ptr(),
				m_rays.ptr(),
				bounce, rrBounce,
				m_shapeparam.ptr(), m_shapeparam.num(),
				m_mtrlparam.ptr(),
				m_lightparam.ptr(), m_lightparam.num(),
				m_nodetex.ptr(),
				m_primparams.ptr(),
				texVtxPos, texVtxNml,
				m_mtxparams.ptr(),
				m_tex.ptr());
		}

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
			m_paths.ptr(),
			width, height);
	}
}
