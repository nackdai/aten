#include "camera/thinlens.h"

namespace aten {
	// NOTE
	// https://www.slideshare.net/h013/edubpt-v100

	static real plane_intersection(const vec3& normal, const vec3& pos, const ray& ray)
	{
		/*
			レイの方程式 : x = p + t v
			平面の方程式 : (p - p0)・n = 0
			p  : レイの出る点
			v  : レイの方向
			p0 : 平面上の１点
			n  : 法線ベクトル

			連立した結果、レイの進んだ距離tは

			(p0 - p)・n
			t = ------------
			v・n

			v・n=0 の時は、レイと平面が平行に進んでいるので、交わらない.
			t<0の時は、視線の後ろ側に交点があるので、やはり交わることがない.
		*/

		real vn = dot(ray.dir, normal);

		if (fabs(vn) > AT_MATH_EPSILON) {
			real xpn = dot(pos - ray.org, normal);
			real t = xpn / vn;
			return t;
		}

		return AT_MATH_INF;
	}

	// NOTE
	// xI : イメージセンサー上の点.
	// x0 : レンズ上の点.
	// xV : オブジェクトプレーン上の点.
	// x1 : シーン内の点.

	void ThinLensCamera::init(
		int width, int height,
		vec3 lookfrom, vec3 lookat, vec3 vup,
		real imageSensorSize,
		real imageSensorToLensDistance,
		real lensToObjectplaneDistance,	// focus length
		real lensRadius,
		real W_scale)
	{
		m_imageWidthPx = width;
		m_imageHeightPx = height;

		m_imageSensorToLensDistance = imageSensorToLensDistance;
		m_lensToObjectplaneDistance = lensToObjectplaneDistance;

		// イメージセンサのサイズ.
		m_imagesensor.width = imageSensorSize * m_imageWidthPx / m_imageHeightPx;
		m_imagesensor.height = imageSensorSize;

		// １ピクセルの物理サイズ.
		m_pixelWidth = m_imagesensor.width / m_imageWidthPx;
		m_pixelHeight = m_imagesensor.height / m_imageHeightPx;

		m_imagesensor.center = lookfrom;
		m_imagesensor.dir = normalize(lookat - lookfrom);
		m_imagesensor.u = normalize(cross(m_imagesensor.dir, vup)) * m_imagesensor.width;
		m_imagesensor.v = normalize(cross(m_imagesensor.u, m_imagesensor.dir)) * m_imagesensor.height;

		m_imagesensor.lower_left = m_imagesensor.center - 0.5f * m_imagesensor.u - 0.5f * m_imagesensor.v;

		// オブジェクトプレーンはイメージセンサーと平行なので、イメージセンサーと同じ方向を使えばいい.
		m_objectplane.center = m_imagesensor.center + (m_imageSensorToLensDistance + m_lensToObjectplaneDistance) * m_imagesensor.dir;
		m_objectplane.normal = m_imagesensor.dir;
		m_objectplane.u = m_imagesensor.u;
		m_objectplane.v = m_imagesensor.v;

		m_objectplane.lower_left = m_objectplane.center - 0.5f * m_objectplane.u - 0.5f * m_objectplane.v;

		// レンズはイメージセンサーと平行なので、イメージセンサーと同じ方向を使えばいい.
		m_lens.center = m_imagesensor.center + m_imageSensorToLensDistance * m_imagesensor.dir;
		m_lens.u = lensRadius * normalize(m_imagesensor.u);
		m_lens.v = lensRadius * normalize(m_imagesensor.v);
		m_lens.normal = m_imagesensor.dir;
		m_lens.radius = lensRadius;

		// W(xI <- x0) センサのセンシティビティは簡単のため定数にしておく.
		m_W = W_scale / (m_pixelWidth * m_pixelHeight);
	}

	CameraSampleResult ThinLensCamera::sample(
		real s, real t,
		sampler* sampler) const
	{
		AT_ASSERT(sampler);

		// [0, 1] -> [-0.5, 0.5]
		s -= 0.5;
		t -= 0.5;

		CameraSampleResult result;

		// イメージセンサ上の座標、[-0,5, 0.5]の範囲（0,0)がイメージセンサ中央を示す.
		result.posOnImageSensor = m_imagesensor.center + s * m_imagesensor.u + t * m_imagesensor.v;

		// オブジェクトプレーン上の座標計算
		// オブジェクトプレーンのサイズは、レンズの公式の倍率計算（m=b/a）
		real ratio = m_lensToObjectplaneDistance / m_imageSensorToLensDistance;

		// センサーとオブジェクトプレーンの向きは反対になるので、マイナスする?
		real u_on_objectplane = -ratio * s;
		real v_on_objectplane = -ratio * t;
		result.posOnObjectplane = m_objectplane.center + u_on_objectplane * m_objectplane.u + v_on_objectplane * m_objectplane.v;

		// NOTE
		// lens.u、lens.v に lens.radius が含まれているので、レンズの半径について考慮する必要がない.
		real r0 = sqrt(sampler->nextSample());
		real r1 = sampler->nextSample() * real(2.0) * AT_MATH_PI;
		real u = r0 * cos(r1);
		real v = r0 * sin(r1);
		result.posOnLens = m_lens.center + u * m_lens.u + v * m_lens.v;
		result.nmlOnLens = m_lens.normal;

		// ピクセル内の一点をサンプリングする確率密度関数（面積測度）
		result.pdfOnImageSensor = real(1.0) / (m_pixelWidth * m_pixelHeight);

		// レンズ上の一点をサンプリングする確率密度関数（面積測度）
		result.pdfOnLens = real(1.0) / (AT_MATH_PI * m_lens.radius * m_lens.radius);

		result.r = ray(
			result.posOnLens,
			normalize(result.posOnObjectplane - result.posOnLens));

		return std::move(result);
	}

	real ThinLensCamera::hitOnLens(
		const ray& r,
		vec3& posOnLens,
		vec3& posOnObjectPlane,
		vec3& posOnImageSensor,
		int& x, int& y) const
	{
		// レンズと判定
		auto lens_t = plane_intersection(m_lens.normal, m_lens.center, r);

		if (AT_MATH_EPSILON < lens_t)
		{
			posOnLens = r.org + lens_t * r.dir;
			auto l = (posOnLens - m_lens.center).length();
			auto d = dot(m_lens.normal, r.dir);

			if (l < m_lens.radius && d <= 0.0)
			{
				auto objplane_t = plane_intersection(m_objectplane.normal, m_objectplane.center, r);

				posOnObjectPlane = r.org + objplane_t * r.dir;

				auto u_on_objectplane = dot(posOnObjectPlane - m_objectplane.center, normalize(m_objectplane.u)) / m_objectplane.u.length();
				auto v_on_objectplane = dot(posOnObjectPlane - m_objectplane.center, normalize(m_objectplane.v)) / m_objectplane.v.length();

				auto ratio = m_imageSensorToLensDistance / m_lensToObjectplaneDistance;
				auto u = -ratio * u_on_objectplane;
				auto v = -ratio * v_on_objectplane;

				posOnImageSensor = m_imagesensor.center + u * m_imagesensor.u + v * m_imagesensor.v;

				if (-0.5f <= u && u < 0.5
					&& -0.5f <= v && v < 0.5)
				{
					x = (int)((u + 0.5f) * m_imageWidthPx);
					y = (int)((v + 0.5f) * m_imageHeightPx);

					return lens_t;
				}
			}
		}

		return -AT_MATH_INF;
	}

	// イメージセンサ上のサンプリング確率密度（イメージセンサの面積測度に関する確率密度）をシーン上のサンプリング確率密度（面積測度に関する確率密度）に変換する.
	real ThinLensCamera::convertImageSensorPdfToScenePdf(
		real pdfImage,
		const vec3& hitPoint,
		const vec3& hitpointNml,
		const vec3& posOnImageSensor,
		const vec3& posOnLens,
		const vec3& posOnObjectPlane) const
	{
		// NOTE
		// http://rayspace.xyz/CG/contents/DoF.html

		// NOTE
		// https://www.slideshare.net/h013/edubpt-v100
		// イメージセンサ上の確立密度を面積測度に変換.
		// p111
		// Pa = (B/A)^2 * (r''/r)^2 * (cosΘ'/cosΘ'') * Pij

		const vec3& x0 = posOnLens;
		const vec3& xI = posOnImageSensor;
		const vec3& xV = posOnObjectPlane;
		const vec3& x1 = hitPoint;

		const vec3 x0_xV = xV - x0;
		const vec3 x0_x1 = x1 - x0;

		// NOTE
		// B = imagesensor_to_lens_distance
		// A = lens_to_objectplane(focus distance)

		// (B/A)^2
		const real ba = m_imageSensorToLensDistance / m_lensToObjectplaneDistance;
		const real ba2 = ba * ba;

		// (r''/r')^2
		auto r2 = (x0_xV.squared_length() / x0_x1.squared_length());

		// (cosΘ'/cosΘ'')
		auto c2 = (dot(normalize(-x0_x1), hitpointNml) / dot(normalize(x0_x1), normalize(m_imagesensor.dir)));

		// イメージセンサの面積測度に関する確率密度.
		//auto pdfOnImageSensor = 1.0 / (m_imagesensor.width * m_imagesensor.height);

		auto pdf = pdfImage * ba2 * r2 * c2;

		return pdf;
	}

	real ThinLensCamera::getSensitivity(
		const vec3& posOnImagesensor,
		const vec3& posOnLens) const
	{
		// NOTE
		// https://www.slideshare.net/h013/edubpt-v100
		// p54 - p62

		const vec3 x0_xI = posOnImagesensor - posOnLens;

		// センサ上の点の半球積分を、レンズ上の点の積分に変数変換した時に導入される係数
		// (cosΘ)^2/r^2
		const real cos = dot(normalize(-m_imagesensor.dir), normalize(x0_xI));
		const real len2 = x0_xI.squared_length();
		
		const real G = (cos * cos) / len2;

		real ret = m_W * G;

		return ret;
	}

	real ThinLensCamera::getWdash(
		const vec3& hitPoint,
		const vec3& hitpointNml,
		const vec3& posOnImageSensor,
		const vec3& posOnLens,
		const vec3& posOnObjectPlane) const
	{
		// 幾何的な係数計算 + センサーセンシティビティの項を計算する.
		// x1 -> x0への放射輝度が最終的にイメージセンサに与える寄与度.

		const vec3& x0 = posOnLens;
		const vec3& xI = posOnImageSensor;
		const vec3& xV = posOnObjectPlane;
		const vec3& x1 = hitPoint;

		const vec3 x0_xI = xI - x0;	// lens to image sensor.
		const vec3 x0_xV = xV - x0;	// lens to object plane.
		const vec3 x0_x1 = x1 - x0;	// lens to hit point.

		// NOTE
		// B = imagesensor_to_lens_distance
		// A = lens_to_objectplane(focus distance)

#if 1
		// B/A
		const real ba2 = x0_xV.squared_length() / x0_xI.squared_length();

		const real r = m_imageSensorToLensDistance / m_lensToObjectplaneDistance;
		const real r2 = r * r;

		const real c = dot(normalize(x0_xI), normalize(-m_imagesensor.dir)) / dot(normalize(x0_x1), normalize(m_imagesensor.dir));
		const real c2 = c * c;

		real W_dash = m_W * ba2 * r2 * c2;
#else
		// B/A
		const real ba = m_imageSensorToLensDistance / m_lensToObjectplaneDistance;
		const real ba2 = ba * ba;

		const real d0 = x0_xI.length();
		const real c0 = dot(normalize(x0_xI), normalize(-m_imagesensor.dir));
		const real G0 = c0 * c0 / (d0 * d0);

		const real d1 = x0_xV.length();
		const real c1 = dot(normalize(x0_x1), normalize(m_imagesensor.dir));
		const real G1 = c1 * c1 / (d1 * d1);

		real W_dash = m_W * ba2 / G0 * G1;
#endif

		return W_dash;
	}

	void ThinLensCamera::revertRayToPixelPos(
		const ray& ray,
		int& px, int& py) const
	{
		vec3 posOnLens;
		vec3 posOnObjectPlane;
		vec3 posOnImageSensor;

		hitOnLens(
			ray,
			posOnLens,
			posOnObjectPlane,
			posOnImageSensor,
			px, py);
	}
}
