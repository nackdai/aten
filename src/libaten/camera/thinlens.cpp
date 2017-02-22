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

	void ThinLensCamera::init(
		int width, int height,
		vec3 lookfrom, vec3 lookat, vec3 vup,
		real imageSensorSize,
		real imageSensorToLensDistance,
		real lensToObjectplaneDistance,
		real lensRadius,
		real W_scale)
	{
		m_imageWidthPx = width;
		m_imageHeightPx = height;

		m_imageSensorToLensDistance = imageSensorToLensDistance;
		m_lensToObjectplaneDistance = lensToObjectplaneDistance;

		m_imagesensor.width = imageSensorSize * m_imageWidthPx / m_imageHeightPx;
		m_imagesensor.height = imageSensorSize;

		m_pixelWidth = m_imagesensor.width / m_imageWidthPx;
		m_pixelHeight = m_imagesensor.height / m_imageHeightPx;

		m_imagesensor.center = lookfrom;
		m_imagesensor.dir = normalize(lookat - lookfrom);
		m_imagesensor.u = normalize(cross(m_imagesensor.dir, vup)) * m_imagesensor.width;
		m_imagesensor.v = normalize(cross(m_imagesensor.u, m_imagesensor.dir)) * m_imagesensor.height;

		m_imagesensor.lower_left = m_imagesensor.center - 0.5f * m_imagesensor.u - 0.5f * m_imagesensor.v;

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
		real r1 = sampler->nextSample() * 2.0 * AT_MATH_PI;
		real u = r0 * cos(r1);
		real v = r0 * sin(r1);
		result.posOnLens = m_lens.center + u * m_lens.u + v * m_lens.v;

		// ピクセル内の一点をサンプリングする確率密度関数（面積測度）
		result.pdfOnImageSensor = 1.0 / (m_pixelWidth * m_pixelHeight);

		// レンズ上の一点をサンプリングする確率密度関数（面積測度）
		result.pdfOnLens = 1.0 / (AT_MATH_PI * m_lens.radius * m_lens.radius);

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

		return AT_MATH_INF;
	}

	real ThinLensCamera::getPdfImageSensorArea(
		const vec3& hitpoint,
		const vec3& hitpointNml) const
	{
		// NOTE
		// http://rayspace.xyz/CG/contents/DoF.html

		// NOTE
		// https://www.slideshare.net/h013/edubpt-v100
		// イメージセンサ上の確立密度を面積測度に変換.
		// p111
		// Pa = (B/A)^2 * (r''/r)^2 * (cosΘ'/cosΘ'')Pi

		const vec3 w_x0_x1 = hitpoint - m_lens.center;
		const vec3& n0 = m_lens.normal;
		const vec3& n1 = hitpointNml;

		const real wdotn0 = aten::abs(dot(normalize(w_x0_x1), n0));
		const real wdotn1 = aten::abs(dot(normalize(w_x0_x1), n1));
		const real dp2 = m_imageSensorToLensDistance * m_imageSensorToLensDistance;
		const real dist2 = w_x0_x1.squared_length();

		// ピクセル内の一点をサンプリングする確率密度関数（面積測度）
		auto pdfOnImageSensor = 1.0 / (m_pixelWidth * m_pixelHeight);

		auto pdf = (wdotn1 * dp2) / (dist2 * wdotn0 * wdotn0 * wdotn0) * pdfOnImageSensor;

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
		const vec3& posOnImageSensor,
		const vec3& posOnLens,
		const vec3& posOnObjectPlane) const
	{
		// 幾何的な係数計算 + センサーセンシティビティの項を計算する.
		// x1 -> x0への放射輝度が最終的にイメージセンサに与える寄与度.

		// lens <-> image sensor
		vec3 x0_xI = posOnImageSensor - posOnLens;

		// lens <-> object plane 
		vec3 x0_xV = posOnObjectPlane - posOnLens;

		real ba = m_imageSensorToLensDistance / m_lensToObjectplaneDistance;  // B /A

		real r = x0_xI.length();          // r   => lens <-> image sensor
		real r_2dash = x0_xV.length();    // r'' => lens <-> object plane 

		// lens <-> image sensor
		real cos_theta = dot(normalize(x0_xI), -m_imagesensor.dir);

		// lens <-> object plane 
		real cos_theta_2dash = dot(normalize(x0_xV), m_imagesensor.dir);

		real W_dash = m_W * pow(ba * (r_2dash / r) * (cos_theta / cos_theta_2dash), 2.0);

		return W_dash;
	}
}
