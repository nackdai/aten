#include "accelerator/CullingFrusta.h"

namespace aten {
	void CullingFrusta::init(
		uint32_t width, uint32_t height,
		uint32_t tileSize,
		const camera* cam)
	{
		// TODO
		AT_ASSERT(width % tileSize == 0);
		AT_ASSERT(height % tileSize == 0);

		const auto& camorg = cam->getPos();

		m_xTileNum = width / tileSize;
		m_yTileNum = height / tileSize;

		m_tiles.clear();
		m_tiles.resize(m_xTileNum * m_yTileNum);

		m_width = width;
		m_height = height;
		m_tileSize = tileSize;

		for (int y = 0; y < m_yTileNum; y++) {
			for (int x = 0; x < m_xTileNum; x++) {
				int pos = y * m_xTileNum + x;

				auto& tile = m_tiles[pos];

				tile.x0 = x * tileSize;
				tile.y0 = y * tileSize;
				tile.x1 = tile.x0 + tileSize;
				tile.y1 = tile.y0 + tileSize;

				updateTile(cam, camorg, tile);
			}
		}
	}

	void CullingFrusta::updateTile(
		const camera* cam,
		const vec3& camorg,
		Tile& tile)
	{
		// NOTE
		// 0--3
		// |  |
		// 1--2
		// 0 : (x0, y0)
		// 2 : (x1, y1)

		vec3 p[4] = {
			vec3(tile.x0, tile.y0, real(0)),
			vec3(tile.x0, tile.y0 + m_tileSize, real(0)),
			vec3(tile.x1, tile.y1, real(0)),
			vec3(tile.x0 + m_tileSize, tile.y0, real(0)),
		};

		for (int i = 0; i < 4; i++) {
			p[i].x /= (real)m_width;
			p[i].y /= (real)m_height;

			// TODO
			// Only for Pinhole now...
			auto camsample = cam->sample(p[i].x, p[i].y, nullptr);

			p[i] = camsample.r.dir;
		}

		tile.f.update(camorg, p);
	}

	void CullingFrusta::update(const camera* cam)
	{
		const auto& camorg = cam->getPos();

		for (int y = 0; y < m_yTileNum; y++) {
			for (int x = 0; x < m_xTileNum; x++) {
				int pos = y * m_xTileNum + x;

				auto& tile = m_tiles[pos];

				updateTile(cam, camorg, tile);
			}
		}
	}

	void CullingFrusta::traverseIntersect(
		accelerator* accel,
		std::vector<IntersectResult>& results)
	{
		results.clear();
		results.resize(m_xTileNum * m_yTileNum);

		for (int y = 0; y < m_yTileNum; y++) {
			for (int x = 0; x < m_xTileNum; x++) {
				int pos = y * m_xTileNum + x;

				auto& tile = m_tiles[pos];

				auto res = accel->intersectTestByFrustum(tile.f);

				results[pos].x0 = tile.x0;
				results[pos].y0 = tile.y0;
				results[pos].x1 = tile.x1;
				results[pos].y1 = tile.y1;

				results[pos].isect = res;
			}
		}
	}

	int CullingFrusta::computeTileIdx(int x, int y)
	{
		int xPos = x / m_tileSize;
		int yPos = y / m_tileSize;

		int ret = yPos * m_xTileNum + xPos;

		return ret;
	}
}
