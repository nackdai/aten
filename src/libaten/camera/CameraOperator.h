#pragma once

#include "types.h"
#include "camera/camera.h"

namespace aten {
	class CameraOperator {
	private:
		CameraOperator();
		~CameraOperator();

	public:
		static void move(
			camera& camera,
			real x, real y);

		static void dolly(
			camera& camera,
			real scale);

		static void rotate(
			camera& camera,
			real x1, real y1,
			real x2, real y2);
	};
}