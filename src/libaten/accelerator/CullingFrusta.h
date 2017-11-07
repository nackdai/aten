#pragma once

#include "math/frustum.h"
#include "camera/camera.h"
#include "accelerator/accelerator.h"

#include <vector>

namespace aten {
	class CullingFrusta {
	public:
		CullingFrusta() {}
		~CullingFrusta() {}

	public:
		void init(
			uint32_t width, uint32_t height,
			uint32_t tileSize,
			const camera* cam);

		void update(const camera* cam);

		struct IntersectResult {
			uint32_t x0;
			uint32_t y0;
			uint32_t x1;
			uint32_t y1;

			accelerator::ResultIntersectTestByFrustum isect;
		};

		void traverseIntersect(
			accelerator* accel,
			std::vector<IntersectResult>& results);

		int computeTileIdx(int x, int y);

		const frustum& getFrustum(uint32_t idx) const
		{
			return m_tiles[idx].f;
		}

	private:
		struct Tile {
			uint32_t x0;
			uint32_t y0;
			uint32_t x1;
			uint32_t y1;

			frustum f;
		};

		void updateTile(
			const camera* cam,
			const vec3& camorg,
			Tile& tile);

	private:
		std::vector<Tile> m_tiles;

		uint32_t m_width;
		uint32_t m_height;
		uint32_t m_tileSize;

		int m_xTileNum;
		int m_yTileNum;
	};
}
