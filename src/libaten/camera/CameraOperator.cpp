#include "camera/CameraOperator.h"
#include "math/vec4.h"
#include "math/mat4.h"

#pragma optimize( "", off)

namespace aten {
	void CameraOperator::move(
		camera& camera,
		real x, real y)
	{
		auto& pos = camera.getPos();
		auto& at = camera.getAt();

		// 移動ベクトル.
		aten::vec3 offset(x, y, real(0));

		// カメラの回転を考慮する.
		aten::vec3 dir = at - pos;
		dir = normalize(dir);
		dir.y = real(0);

		aten::mat4 mtxRot;
		mtxRot.asRotateFromVector(dir, aten::vec3(0, 1, 0));
		
		mtxRot.applyXYZ(offset);

		pos += offset;
		at += offset;
	}

	void CameraOperator::dolly(
		camera& camera,
		real scale)
	{
		auto& pos = camera.getPos();
		auto& at = camera.getAt();

		// 視点と注視点の距離.
		real len = length(pos - at);

		// 視点から注視点への方向.
		auto dir = pos - at;
		dir = normalize(dir);

		// スケーリング.
		// 注視点までの距離に応じる.
		auto distScale = scale * len * 0.01f;
		dir *= distScale;

		// 新しい視点.
		pos += dir;
	}

	static inline real projectionToSphere(
		real radius,
		real x,
		real y)
	{
		real z = real(0);
		real dist = aten::sqrt(x * x + y * y);

		// r * 1/√2 の点で双曲線と接する内側と外側

		if (dist < radius * real(0.70710678118654752440)) {
			// 内側

			// NOTE
			// r * r = x * x + y * y + z * z
			// <=> z * z = r * r - (x * x + y * y)
			z = aten::sqrt(radius * radius - dist * dist);
		}
		else {
			// 外側
			real t = radius * radius * 0.5f;
			z = t / dist;
		}

		return z;
	}

	void CameraOperator::rotate(
		camera& camera,
		real x1, real y1,
		real x2, real y2)
	{
		static const real radius = real(0.8);

		// スクリーン上の２点からトラックボール上の点を計算する.
		// GLUTと同じ方法.

		aten::vec3 v1(
			x1, y1,
			projectionToSphere(radius, x1, y1));
		v1 = normalize(v1);

		aten::vec3 v2(
			x2, y2,
			projectionToSphere(radius, x2, y2));
		v2 = normalize(v2);

		// 回転軸.
		auto axis = cross(v1, v2);
		axis = normalize(axis);

		const auto dir = camera.getDir();
		aten::mat4 transform;
		transform.asRotateFromVector(dir, aten::vec3(0, 1, 0));

		// カメラの回転状態に合わせて軸も回転.
		transform.applyXYZ(axis);

		// 回転の角度
		// NOTE
		// V1・V2 = |V1||V2|cosθ = cosθ (|V1| = |V2| = 1)
		// θ = acos(cosθ)
		// => θ = acos(cosθ) = acos(V1・V2)
		real theta = aten::acos(dot(v1, v2));

		// 回転.
		aten::mat4 mtxRot;
		mtxRot.asRotateByAxis(theta, axis);

		auto& pos = camera.getPos();
		auto& at = camera.getAt();

		pos -= at;
		pos = mtxRot.applyXYZ(pos);
		pos += at;
	}
}