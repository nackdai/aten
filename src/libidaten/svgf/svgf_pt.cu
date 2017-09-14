#include "svgf/svgf_pt.h"

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

__global__ void hitTest(
	idaten::SVGFPathTracing::Path* paths,
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
#if 0
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= width * height) {
		return;
	}
#else
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);
#endif

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
}

template <bool isFirstBounce>
__global__ void shadeMiss(
	idaten::SVGFPathTracing::AOV* aovs,
	idaten::SVGFPathTracing::Path* paths,
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

			// Export bg color to albedo buffer.
			aovs[idx].texclr = make_float4(bg.x, bg.y, bg.z, 1);
			aovs[idx].depth = -1;
			aovs[idx].meshid = -1;
			aovs[idx].mtrlid = -1;

			// For exporting separated albedo.
			bg = aten::vec3(1, 1, 1);
		}

		path.contrib += path.throughput * bg;

		path.isTerminate = true;
	}
}

template <bool isFirstBounce>
__global__ void shadeMissWithEnvmap(
	idaten::SVGFPathTracing::AOV* aovs,
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

			// Export envmap to albedo buffer.
			aovs[idx].texclr = make_float4(emit.x, emit.y, emit.z, 1);
			aovs[idx].depth = -1;
			aovs[idx].meshid = -1;
			aovs[idx].mtrlid = -1;

			// For exporting separated albedo.
			emit = aten::vec3(1, 1, 1);
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

template <bool isFirstBounce>
__global__ void shade(
	idaten::SVGFPathTracing::AOV* aovs,
	aten::mat4 mtxW2C,
	int width, int height,
	idaten::SVGFPathTracing::Path* paths,
	const int* __restrict__ hitindices,
	int hitnum,
	const aten::Intersection* __restrict__ isects,
	aten::ray* rays,
	int frame,
	int bounce, int rrBounce,
	const aten::ShapeParameter* __restrict__ shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	const aten::LightParameter* __restrict__ lights, int lightnum,
	cudaTextureObject_t* nodes,
	const aten::PrimitiveParamter* __restrict__ prims,
	cudaTextureObject_t vtxPos,
	cudaTextureObject_t vtxNml,
	const aten::mat4* __restrict__ matrices,
	cudaTextureObject_t* textures,
	unsigned int* random)
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

	auto& path = paths[idx];
	const auto& ray = rays[idx];

#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
	auto scramble = random[idx] * 0x1fe3434f;
	path.sampler.init(frame, 4 + bounce * 300, scramble);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
	auto rnd = random[idx];
	auto scramble = rnd * 0x1fe3434f * ((frame + 331 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
	path.sampler.init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 4 + bounce * 300, scramble);
#endif

	aten::hitrecord rec;

	const auto& isect = isects[idx];

	auto obj = &ctxt.shapes[isect.objid];
	evalHitResult(&ctxt, obj, ray, &rec, &isect);

	aten::MaterialParameter mtrl = ctxt.mtrls[rec.mtrlid];

	bool isBackfacing = dot(rec.normal, -ray.dir) < 0.0f;

	// 交差位置の法線.
	// 物体からのレイの入出を考慮.
	aten::vec3 orienting_normal = rec.normal;

	if (mtrl.type != aten::MaterialType::Layer) {
		mtrl.albedoMap = (int)(mtrl.albedoMap >= 0 ? ctxt.textures[mtrl.albedoMap] : -1);
		mtrl.normalMap = (int)(mtrl.normalMap >= 0 ? ctxt.textures[mtrl.normalMap] : -1);
		mtrl.roughnessMap = (int)(mtrl.roughnessMap >= 0 ? ctxt.textures[mtrl.roughnessMap] : -1);
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

		// normal
		aovs[idx].normal = make_float4(orienting_normal.x, orienting_normal.y, orienting_normal.z, 0);

		// depth, meshid.
		aovs[idx].depth = pos.w;
		aovs[idx].meshid = isect.meshid;
		aovs[idx].mtrlid = rec.mtrlid;

		// texture color.
		auto texcolor = AT_NAME::material::sampleTexture(mtrl.albedoMap, rec.u, rec.v, 1.0f);
		aovs[idx].texclr = make_float4(texcolor.x, texcolor.y, texcolor.z, 1);

		// For exporting separated albedo.
		mtrl.albedoMap = -1;
	}

	// Implicit conection to light.
	if (mtrl.attrib.isEmissive) {
		if (!isBackfacing) {
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
		}

		// When ray hit the light, tracing will finish.
		path.isTerminate = true;
		return;
	}

	if (!mtrl.attrib.isTranslucent && isBackfacing) {
		orienting_normal = -orienting_normal;
	}

	// Apply normal map.
	int normalMap = mtrl.normalMap;
	if (mtrl.type == aten::MaterialType::Layer) {
		// 最表層の NormalMap を適用.
		auto* topmtrl = &ctxt.mtrls[mtrl.layer[0]];
		normalMap = (int)(topmtrl->normalMap >= 0 ? ctxt.textures[topmtrl->normalMap] : -1);
	}
	AT_NAME::material::applyNormalMap(normalMap, orienting_normal, orienting_normal, rec.u, rec.v);

#if 1
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
		
		auto shadowRayOrg = rec.p + AT_MATH_EPSILON * orienting_normal;
		auto tmp = rec.p + dirToLight - shadowRayOrg;
		auto shadowRayDir = normalize(tmp);
		aten::ray shadowRay(shadowRayOrg, shadowRayDir);

		bool isHit = intersectCloserBVH(&ctxt, shadowRay, &isectTmp, distToLight - AT_MATH_EPSILON);

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
#endif

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

__global__ void gather(
	cudaSurfaceObject_t dst,
	idaten::SVGFPathTracing::AOV* aovs,
	const idaten::SVGFPathTracing::Path* __restrict__ paths,
	int width, int height)
{
	auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

#ifdef ENABLE_DEBUG_1PIXEL
	ix = DEBUG_IX;
	iy = DEBUG_IY;
#endif

	const auto idx = getIdx(ix, iy, width);

	const auto& path = paths[idx];

	int sample = path.samples;

	float4 contrib = make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 0) / sample;
	contrib.w = sample;

	float lum = AT_NAME::color::luminance(contrib.x, contrib.y, contrib.z);

	aovs[idx].moments += make_float4(lum * lum, lum, 0, 1);

	aovs[idx].color = contrib;

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
		contrib,
		dst,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
#endif
}

namespace idaten
{
	void SVGFPathTracing::update(
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

		auto& r = aten::getRandom();
		m_random.init(width * height);
		m_random.writeByNum(&r[0], width * height);

		for (int i = 0; i < 2; i++) {
			m_aovs[i].init(width * height);
		}

		for (int i = 0; i < 2; i++) {
			m_atrousClr[i].init(width * height);
			m_atrousVar[i].init(width * height);
		}

		m_tmpBuf.init(width * height);
	}

	static bool doneSetStackSize = false;

	void SVGFPathTracing::render(
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

		static const int rrBounce = 3;

		// Set bounce count to 1 forcibly, aov render mode.
		maxBounce = (m_mode == Mode::AOVar ? 1 : maxBounce);

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

		if (m_mode == Mode::SVGF)
		{
			onVarianceEstimation(outputSurf, width, height);

			onAtrousFilter(outputSurf, width, height);

			copyFromTmpBufferToAov(width, height);
		}
		else if (m_mode == Mode::VAR) {
			onVarianceEstimation(outputSurf, width, height);
		}

		pick(
			m_pickedInfo.ix, m_pickedInfo.iy, 
			width, height,
			vtxTexPos);

		checkCudaErrors(cudaDeviceSynchronize());

		// Toggle aov buffer pos.
		m_curAOVPos = 1 - m_curAOVPos;

		m_frame++;

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
		}
	}

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
		cudaTextureObject_t texVtxPos)
	{
#if 0
		dim3 blockPerGrid_HitTest((width * height + 128 - 1) / 128);
		dim3 threadPerBlock_HitTest(128);

		hitTest << <blockPerGrid_HitTest, threadPerBlock_HitTest >> > (
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
			m_mtxparams.ptr());

		checkCudaKernel(hitTest);
	}

	void SVGFPathTracing::onShadeMiss(
		int width, int height,
		int bounce)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		auto& curaov = getCurAovs();

		if (m_envmapRsc.idx >= 0) {
			if (bounce == 0) {
				shadeMissWithEnvmap<true> << <grid, block >> > (
					curaov.ptr(),
					m_tex.ptr(),
					m_envmapRsc.idx, m_envmapRsc.avgIllum, m_envmapRsc.multiplyer,
					m_paths.ptr(),
					m_rays.ptr(),
					width, height);
			}
			else {
				shadeMissWithEnvmap<false> << <grid, block >> > (
					curaov.ptr(),
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
					curaov.ptr(),
					m_paths.ptr(),
					width, height);
			}
			else {
				shadeMiss<false> << <grid, block >> > (
					curaov.ptr(),
					m_paths.ptr(),
					width, height);
			}
		}

		checkCudaKernel(shadeMiss);
	}

	void SVGFPathTracing::onShade(
		cudaSurfaceObject_t outputSurf,
		int hitcount,
		int width, int height,
		int bounce, int rrBounce,
		cudaTextureObject_t texVtxPos,
		cudaTextureObject_t texVtxNml)
	{
		aten::mat4 mtxW2V;
		mtxW2V.lookat(
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

		aten::mat4 mtxW2C = m_mtxV2C * mtxW2V;

#ifdef ENABLE_DEBUG_1PIXEL
		int blockPerGrid = 1;
		int threadPerBlock = 1;
#else
		dim3 blockPerGrid((hitcount + 64 - 1) / 64);
		dim3 threadPerBlock(64);
#endif

		auto& curaov = getCurAovs();

		if (bounce == 0) {
			shade<true> << <blockPerGrid, threadPerBlock >> > (
				curaov.ptr(), mtxW2C,
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
				m_random.ptr());
		}
		else {
			shade<false> << <blockPerGrid, threadPerBlock >> > (
				curaov.ptr(), mtxW2C,
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
				m_random.ptr());
		}

		checkCudaKernel(shade);
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

		if (m_mode == Mode::PT) {
			auto& curaov = getCurAovs();

			gather << <grid, block >> > (
				outputSurf,
				curaov.ptr(),
				m_paths.ptr(),
				width, height);

			checkCudaKernel(gather);
		}
		else if (m_mode == Mode::AOVar) {
			onFillAOV(outputSurf, width, height);
		}
		else {
			auto& curaov = getCurAovs();
			auto& prevaov = getPrevAovs();

			if (isFirstFrame()) {
				gather << <grid, block >> > (
					outputSurf,
					curaov.ptr(),
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

		m_mtxPrevV2C = m_mtxV2C;
	}
}
