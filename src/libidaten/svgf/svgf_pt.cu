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

__global__ void genPath(
	idaten::SVGFPathTracing::Path* paths,
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
	isects[idx].area = isect.area;
	isects[idx].primid = isect.primid;
	isects[idx].a = isect.a;
	isects[idx].b = isect.b;

	path.isHit = isHit;

	hitbools[idx] = isHit ? 1 : 0;
}

template <bool isFirstBounce>
__global__ void shadeMiss(
	cudaSurfaceObject_t* aovs,
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
			surf2Dwrite(
				make_float4(bg.x, bg.y, bg.z, 1),
				aovs[idaten::SVGFPathTracing::AOVType::texclr],
				ix * sizeof(float4), iy,
				cudaBoundaryModeTrap);

			// For exporting separated albedo.
			bg = aten::vec3(1, 1, 1);
		}

		path.contrib += path.throughput * bg;

		path.isTerminate = true;
	}
}

template <bool isFirstBounce>
__global__ void shadeMissWithEnvmap(
	cudaSurfaceObject_t* aovs,
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
			surf2Dwrite(
				make_float4(emit.x, emit.y, emit.z, 1),
				aovs[idaten::SVGFPathTracing::AOVType::texclr],
				ix * sizeof(float4), iy,
				cudaBoundaryModeTrap);

			// TODO
			// 背景だということが分かるように、meshID = -1 をここで出力するか?
			// それとも、texture clear で指定するか?

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
	cudaSurfaceObject_t* aovs,
	aten::mat4 mtxW2C,
	int width, int height,
	idaten::SVGFPathTracing::Path* paths,
	const int* __restrict__ hitindices,
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

	// Render AOVs.
	if (isFirstBounce) {
		int ix = idx % width;
		int iy = idx / width;

		// World coordinate to Clip coordinate.
		aten::vec4 pos = aten::vec4(rec.p, 1);
		pos = mtxW2C.apply(pos);

		// [-1, 1] -> [0, 1]
		auto n = (orienting_normal + 1.0f) * 0.5f;

		// normal
		surf2Dwrite(
			make_float4(n.x, n.y, n.z, 0),
			aovs[idaten::SVGFPathTracing::AOVType::normal],
			ix * sizeof(float4), iy,
			cudaBoundaryModeTrap);

		// depth, meshid.
		surf2Dwrite(
			make_float4(pos.w, isect.meshid, rec.mtrlid, 0),
			aovs[idaten::SVGFPathTracing::AOVType::depth_meshid],
			ix * sizeof(float4), iy,
			cudaBoundaryModeTrap);

		// texture color.

		auto texcolor = AT_NAME::material::sampleTexture(mtrl.albedoMap, rec.u, rec.v, 1.0f);

		surf2Dwrite(
			make_float4(texcolor.x, texcolor.y, texcolor.z, 1),
			aovs[idaten::SVGFPathTracing::AOVType::texclr],
			ix * sizeof(float4), iy,
			cudaBoundaryModeTrap);

		// For exporting separated albedo.
		mtrl.albedoMap = -1;
	}

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
	cudaSurfaceObject_t* aovs,
	const idaten::SVGFPathTracing::Path* __restrict__ paths,
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

	float4 contrib = make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 0) / sample;
	contrib.w = sample;

	float lum = AT_NAME::color::luminance(contrib.x, contrib.y, contrib.z);

	surf2Dwrite(
		make_float4(lum, lum, lum, 1),
		aovs[idaten::SVGFPathTracing::AOVType::lum],
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);

	surf2Dwrite(
		contrib,
		aovs[idaten::SVGFPathTracing::AOVType::clr_history],
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);

	surf2Dwrite(
		contrib,
		dst,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}

inline __device__ void computePrevScreenPos(
	int ix, int iy,
	float centerDepth,
	int width, int height,
	aten::vec4* prevPos,
	const aten::mat4* __restrict__ mtxs)
{
	// NOTE
	// Pview = (Xview, Yview, Zview, 1)
	// mtxV2C = W 0 0  0
	//          0 H 0  0
	//          0 0 A  B
	//          0 0 -1 0
	// mtxV2C * Pview = (Xclip, Yclip, Zclip, Wclip) = (Xclip, Yclip, Zclip, Zview)
	//  Wclip = Zview = depth
	// Xscr = Xclip / Wclip = Xclip / Zview = Xclip / depth
	// Yscr = Yclip / Wclip = Yclip / Zview = Yclip / depth
	//
	// Xscr * depth = Xclip
	// Xview = mtxC2V * Xclip

	const aten::mat4 mtxC2V = mtxs[0];
	const aten::mat4 mtxPrevV2C = mtxs[1];

	float2 uv = make_float2(ix + 0.5, iy + 0.5);
	uv /= make_float2(width - 1, height - 1);	// [0, 1]
	uv = uv * 2.0f - 1.0f;	// [0, 1] -> [-1, 1]

	aten::vec4 pos(uv.x, uv.y, 0, 0);

	// Screen-space -> Clip-space.
	pos.x *= centerDepth;
	pos.y *= centerDepth;

	// Clip-space -> View-space
	pos = mtxC2V.apply(pos);
	pos.z = -centerDepth;
	pos.w = 1.0;

	// Reproject previous screen position.
	*prevPos = mtxPrevV2C.apply(pos);
	*prevPos /= prevPos->w;

	*prevPos = *prevPos * 0.5 + 0.5;	// [-1, 1] -> [0, 1]
}

__global__ void temporalReprojection(
	const idaten::SVGFPathTracing::Path* __restrict__ paths,
	const aten::CameraParameter* __restrict__ camera,
	cudaSurfaceObject_t* curAovs,
	cudaSurfaceObject_t* prevAovs,
	const aten::mat4* __restrict__ mtxs,
	cudaSurfaceObject_t dst,
	int width, int height)
{
	const int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	const auto path = paths[idx];

	float4 depth_meshid;
	surf2Dread(
		&depth_meshid,
		curAovs[idaten::SVGFPathTracing::AOVType::depth_meshid],
		ix * sizeof(float4), iy);

	const float centerDepth = aten::clamp(depth_meshid.x, camera->znear, camera->zfar);
	const int centerMeshId = (int)depth_meshid.y;

	// 今回のフレームのピクセルカラー.
	float4 curColor = make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 0) / path.samples;
	curColor.w = 1;

	if (centerMeshId < 0) {
		// 背景なので、そのまま出力して終わり.
		surf2Dwrite(
			curColor,
			dst,
			ix * sizeof(float4), iy,
			cudaBoundaryModeTrap);

		return;
	}

	float4 centerNormal;
	surf2Dread(
		&centerNormal,
		curAovs[idaten::SVGFPathTracing::AOVType::normal],
		ix * sizeof(float4), iy);

	// [0, 1] -> [-1, 1]
	centerNormal = 2 * centerNormal - 1;
	centerNormal.w = 0;

	float4 sum = make_float4(0, 0, 0, 0);
	float weight = 0.0f;

	float4 prevDepthMeshId;
	float4 prevNormal;

	for (int y = -1; y < 1; y++) {
		for (int x = -1; x < 1; x++) {
			int xx = clamp(ix + x, 0, width - 1);
			int yy = clamp(iy + y, 0, height - 1);

			// 前のフレームのクリップ空間座標を計算.
			aten::vec4 prevPos;
			computePrevScreenPos(
				xx, yy,
				centerDepth,
				width, height,
				&prevPos,
				mtxs);

			// [0, 1]の範囲内に入っているか.
			bool isInsideX = (0.0 <= prevPos.x) && (prevPos.x <= 1.0);
			bool isInsideY = (0.0 <= prevPos.y) && (prevPos.y <= 1.0);

			if (isInsideX && isInsideY) {
				// 前のフレームのスクリーン座標.
				int px = (int)(prevPos.x * width - 0.5f);
				int py = (int)(prevPos.y * height - 0.5f);

				px = clamp(px, 0, width - 1);
				py = clamp(py, 0, height - 1);

				surf2Dread(
					&prevDepthMeshId,
					prevAovs[idaten::SVGFPathTracing::AOVType::depth_meshid],
					px * sizeof(float4), py);

				const float prevDepth = aten::clamp(depth_meshid.x, camera->znear, camera->zfar);
				const int prevMeshId = (int)depth_meshid.y;

				surf2Dread(
					&prevNormal,
					prevAovs[idaten::SVGFPathTracing::AOVType::normal],
					px * sizeof(float4), py);

				// [0, 1] -> [-1, 1]
				prevNormal = 2 * prevNormal - 1;
				prevNormal.w = 0;

				static const float zThreshold = 0.05f;
				static const float nThreshold = 0.98f;

				float Wz = clamp((zThreshold - abs(1 - centerDepth / prevDepth)) / zThreshold, 0.0f, 1.0f);
				float Wn = clamp((dot(centerNormal, prevNormal) - nThreshold) / (1.0f - nThreshold), 0.0f, 1.0f);
				float Wm = centerMeshId == prevMeshId ? 1.0f : 0.0f;

				// 前のフレームのピクセルカラーを取得.
				float4 prev;
				surf2Dread(
					&prev, 
					prevAovs[idaten::SVGFPathTracing::AOVType::clr_history],
					px * sizeof(float4), py);

				float W = Wz * Wn * Wm;
				sum += prev * W;
				weight += W;
			}
		}
	}

	if (weight > 0.0f) {
		sum /= weight;
		curColor = 0.2 * curColor + 0.8 * sum;
	}

	surf2Dwrite(
		curColor,
		curAovs[idaten::SVGFPathTracing::AOVType::clr_history],
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);

	surf2Dwrite(
		curColor,
		dst,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}

namespace idaten\
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

		for (int i = 0; i < 2; i++) {
			m_aovs[i].resize(AOVType::num);

			for (int n = 0; n < (int)AOVType::num; n++) {
				m_aovTex[i].tex[n].init(width, height, 3);
				m_aovTex[i].tex[n].initAsGLTexture();

				m_aovs[i][n].init(
					m_aovTex[i].tex[n].getGLTexHandle(), 
					CudaGLRscRegisterType::ReadWrite);
			}

			m_aovCudaRsc[i].init(AOVType::num);
		}
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

		// Clear textures for aov.
		{
			m_aovTex[m_curAOVPos].tex[AOVType::normal].clearAsGLTexture(aten::vec4(0));
			m_aovTex[m_curAOVPos].tex[AOVType::depth_meshid].clearAsGLTexture(aten::vec4(-1, -1, -1, -1));
			m_aovTex[m_curAOVPos].tex[AOVType::texclr].clearAsGLTexture(aten::vec4(1));
			m_aovTex[m_curAOVPos].tex[AOVType::lum].clearAsGLTexture(aten::vec4(0));

			// TODO
			m_aovTex[m_curAOVPos].tex[AOVType::clr_history].clearAsGLTexture(aten::vec4(0));
			m_aovTex[m_curAOVPos].tex[AOVType::lum_histroy].clearAsGLTexture(aten::vec4(0));
		}

		for (int i = 0; i < 2; i++)
		{
			std::vector<cudaSurfaceObject_t> tmp;

			for (int n = 0; n < m_aovs[i].size(); n++) {
				m_aovs[i][n].map();
				tmp.push_back(m_aovs[i][n].bind());
			}
			m_aovCudaRsc[i].writeByNum(&tmp[0], tmp.size());
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

		// Toggle aov buffer pos.
		m_curAOVPos = 1 - m_curAOVPos;

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

			for (int i = 0; i < 2; i++) {
				for (int n = 0; n < m_aovs[i].size(); n++) {
					m_aovs[i][n].unbind();
					m_aovs[i][n].unmap();
				}
				m_aovCudaRsc[i].reset();
			}
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

	void SVGFPathTracing::onHitTest(
		int width, int height,
		cudaTextureObject_t texVtxPos)
	{
		dim3 blockPerGrid_HitTest((width * height + 128 - 1) / 128);
		dim3 threadPerBlock_HitTest(128);

		hitTest << <blockPerGrid_HitTest, threadPerBlock_HitTest >> > (
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

		dim3 blockPerGrid((hitcount + 64 - 1) / 64);
		dim3 threadPerBlock(64);

		auto& curaov = getCurAovs();

		if (bounce == 0) {
			shade<true> << <blockPerGrid, threadPerBlock >> > (
			//shade<true> << <1, 1 >> > (
				curaov.ptr(), mtxW2C,
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
				curaov.ptr(), mtxW2C,
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
	}

	void SVGFPathTracing::onGather(
		cudaSurfaceObject_t outputSurf,
		int width, int height,
		int maxSamples)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		auto& curaov = getCurAovs();
		auto& prevaov = getPrevAovs();

		if (m_isFirstRender) {
			gather << <grid, block >> > (
				outputSurf,
				curaov.ptr(),
				m_paths.ptr(),
				width, height);
		}
		else {
			aten::mat4 mtxs[2] = {
				m_mtxC2V,
				m_mtxPrevV2C,
			};

			m_mtxs.init(sizeof(aten::mat4) * AT_COUNTOF(mtxs));
			m_mtxs.writeByNum(mtxs, AT_COUNTOF(mtxs));

			temporalReprojection << <grid, block >> > (
				m_paths.ptr(),
				m_cam.ptr(),
				curaov.ptr(),
				prevaov.ptr(),
				m_mtxs.ptr(),
				outputSurf,
				width, height);

			m_mtxs.reset();
		}

		m_mtxPrevV2C = m_mtxV2C;

		m_isFirstRender = false;
	}
}
