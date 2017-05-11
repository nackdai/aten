#include "accelerator/bvh.h"
#include "shape/tranformable.h"
#include "object/object.h"

#include <random>
#include <vector>

#define BVH_SAH
//#define TEST_NODE_LIST

namespace aten {
	int compareX(const void* a, const void* b)
	{
		const hitable* ah = *(hitable**)a;
		const hitable* bh = *(hitable**)b;

		auto left = ah->getBoundingbox();
		auto right = bh->getBoundingbox();

		if (left.minPos().x < right.minPos().x) {
			return -1;
		}
		else {
			return 1;
		}
	}

	int compareY(const void* a, const void* b)
	{
		hitable* ah = *(hitable**)a;
		hitable* bh = *(hitable**)b;

		auto left = ah->getBoundingbox();
		auto right = bh->getBoundingbox();

		if (left.minPos().y < right.minPos().y) {
			return -1;
		}
		else {
			return 1;
		}
	}

	int compareZ(const void* a, const void* b)
	{
		hitable* ah = *(hitable**)a;
		hitable* bh = *(hitable**)b;

		auto left = ah->getBoundingbox();
		auto right = bh->getBoundingbox();

		if (left.minPos().z < right.minPos().z) {
			return -1;
		}
		else {
			return 1;
		}
	}

	void sortList(bvhnode**& list, uint32_t num, int axis)
	{
		switch (axis) {
		case 0:
			::qsort(list, num, sizeof(bvhnode*), compareX);
			break;
		case 1:
			::qsort(list, num, sizeof(bvhnode*), compareY);
			break;
		default:
			::qsort(list, num, sizeof(bvhnode*), compareZ);
			break;
		}
	}

	void bvhnode::build(
		bvhnode** list,
		uint32_t num)
	{
#ifdef BVH_SAH
		bvh::buildBySAH(this, list, num);
#else
		build(list, num, true);
#endif
	}

	void bvhnode::build(
		bvhnode** list,
		uint32_t num,
		bool needSort)
	{
		if (needSort) {
			// TODO
			int axis = (int)(::rand() % 3);

			sortList(list, num, axis);
		}

		if (num == 1) {
			m_left = list[0];
		}
		else if (num == 2) {
			m_left = list[0];
			m_right = list[1];
		}
		else {
			m_left = new bvhnode(list, num / 2);
			m_right = new bvhnode(list + num / 2, num - num / 2);
		}

		if (m_left && m_right) {
			auto boxLeft = m_left->getBoundingbox();
			auto boxRight = m_right->getBoundingbox();

			m_aabb = aabb::merge(boxLeft, boxRight);
		}
		else {
			auto boxLeft = m_left->getBoundingbox();

			m_aabb = boxLeft;
		}
	}

	bool bvhnode::hit(
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec) const
	{
		auto bbox = getBoundingbox();
		auto isHit = bbox.hit(r, t_min, t_max);

		if (isHit) {
			isHit = bvh::hit(this, r, t_min, t_max, rec);
		}

		return isHit;
	}

	///////////////////////////////////////////////////////
#ifdef TEST_NODE_LIST
	static std::vector<BVHNode> snodes;
#endif

	void bvh::build(
		bvhnode** list,
		uint32_t num)
	{
		// TODO
		int axis = (int)(::rand() % 3);

		sortList(list, num, axis);

		m_root = new bvhnode();
#ifdef BVH_SAH
		buildBySAH(m_root, list, num);
#else
		m_root->build(&list[0], num, false);
#endif

#ifdef TEST_NODE_LIST
		if (snodes.empty()) {
			collectNodes(snodes);
		}
#endif
	}

#ifdef TEST_NODE_LIST
	static bool _hit(
		BVHNode* nodes,
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec)
	{
		static const uint32_t stacksize = 64;
		BVHNode* stackbuf[stacksize] = { nullptr };
		BVHNode** stack = &stackbuf[0];

		// push terminator.
		*stack++ = nullptr;

		int stackpos = 1;
		int nestedStackPos = -1;

		auto& shapes = transformable::getShapes();
		auto& prims = face::faces();

		BVHNode* node = &nodes[0];

		aten::ray transformedRay = r;

		bool isNested = false;

		do {
			BVHNode* left = node->left >= 0 ? &nodes[node->left] : nullptr;
			BVHNode* right = node->right >= 0 ? &nodes[node->right] : nullptr;

			bool traverseLeft = false;
			bool traverseRight = false;

			real left_t = AT_MATH_INF;
			real right_t = AT_MATH_INF;

			if (left) {
				if (left->isLeaf()) {
					hitrecord recLeft;
					bool isHitLeft = false;

					if (left->nestid >= 0) {
						traverseLeft = left->bbox.hit(transformedRay, t_min, t_max, &left_t);

						if (traverseLeft) {
							left = &nodes[left->nestid];
							nestedStackPos = isNested ? nestedStackPos : stackpos;
						}
					}
					else {
						if (left->primid >= 0) {
							auto t = shapes[left->shapeid];
							const auto& leftobj = (hitable*)prims[left->primid];
							const auto& param = t->getParam();
							isHitLeft = leftobj->hit(transformedRay, t_min, t_max, recLeft);

							if (isHitLeft) {
								recLeft.p = param.mtxL2W.apply(recLeft.p);
								recLeft.normal = normalize(param.mtxL2W.applyXYZ(recLeft.normal));

								recLeft.obj = (hitable*)t;

								// For test. Not safe...
								auto obj = (object*)t->getHasObject();

								const auto& targetParam = obj->getParam();

								auto f = prims[targetParam.primid];

								const auto& v0 = VertexManager::getVertex(f->param.idx[0]);
								const auto& v1 = VertexManager::getVertex(f->param.idx[1]);

								real orignalLen = 0;
								{
									const auto& p0 = v0.pos;
									const auto& p1 = v1.pos;

									orignalLen = (p1 - p0).length();
								}

								real scaledLen = 0;
								{
									auto p0 = param.mtxL2W.apply(v0.pos);
									auto p1 = param.mtxL2W.apply(v1.pos);

									scaledLen = (p1 - p0).length();
								}

								real ratio = scaledLen / orignalLen;
								ratio = ratio * ratio;

								recLeft.area = targetParam.area * ratio;
							}
						}
						else {
							const auto& leftobj = (hitable*)shapes[left->shapeid];
							isHitLeft = leftobj->hit(transformedRay, t_min, t_max, recLeft);
						}
					}
					if (isHitLeft) {
						if (recLeft.t < rec.t) {
							rec = recLeft;
						}
					}
				}
				else {
					traverseLeft = left->bbox.hit(transformedRay, t_min, t_max);
				}
			}
			if (right) {
				if (right->isLeaf()) {
					hitrecord recRight;
					bool isHitRight = false;

					if (right->nestid >= 0) {
						traverseRight = right->bbox.hit(transformedRay, t_min, t_max, &right_t);

						if (traverseRight && right_t < left_t) {
							right = &nodes[right->nestid];
							nestedStackPos = isNested ? nestedStackPos : stackpos;
						}
					}
					else {
						if (right->primid >= 0) {
							auto t = shapes[right->shapeid];
							const auto& param = t->getParam();
							const auto rightobj = (hitable*)prims[right->primid];
							isHitRight = rightobj->hit(transformedRay, t_min, t_max, recRight);

							if (isHitRight) {
								recRight.p = param.mtxL2W.apply(recRight.p);
								recRight.normal = normalize(param.mtxL2W.applyXYZ(recRight.normal));

								recRight.obj = (hitable*)t;

								// For test. Not safe...
								auto obj = (object*)t->getHasObject();

								const auto& targetParam = obj->getParam();

								auto f = prims[targetParam.primid];

								const auto& v0 = VertexManager::getVertex(f->param.idx[0]);
								const auto& v1 = VertexManager::getVertex(f->param.idx[1]);

								real orignalLen = 0;
								{
									const auto& p0 = v0.pos;
									const auto& p1 = v1.pos;

									orignalLen = (p1 - p0).length();
								}

								real scaledLen = 0;
								{
									auto p0 = param.mtxL2W.apply(v0.pos);
									auto p1 = param.mtxL2W.apply(v1.pos);

									scaledLen = (p1 - p0).length();
								}

								real ratio = scaledLen / orignalLen;
								ratio = ratio * ratio;

								recRight.area = targetParam.area * ratio;
							}
						}
						else {
							const auto rightobj = (hitable*)shapes[right->shapeid];
							isHitRight = rightobj->hit(transformedRay, t_min, t_max, recRight);
						}
					}
					if (isHitRight) {
						if (recRight.t < rec.t) {
							rec = recRight;
						}
					}
				}
				else {
					traverseRight = right->bbox.hit(transformedRay, t_min, t_max);
				}
			}

			if (!traverseLeft && !traverseRight) {
				if (nestedStackPos == stackpos) {
					nestedStackPos = -1;
					isNested = false;
					transformedRay = r;
				}

				node = *(--stack);
				stackpos -= 1;
			}
			else {
				node = traverseLeft ? left : right;

				if (traverseLeft && traverseRight) {
					*(stack++) = right;
					stackpos += 1;
				}

				if (!isNested && nestedStackPos >= 0) {
					auto t = shapes[node->shapeid];
					const auto& param = t->getParam();
					transformedRay = param.mtxW2L.applyRay(r);
					isNested = true;
				}
			}
			AT_ASSERT(0 <= stackpos && stackpos < stacksize);

			if (stackpos >= stacksize) {
				return false;
			}
		} while (node != nullptr);

		return (rec.obj != nullptr);
	}
#endif

	bool bvh::hit(
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec) const
	{
#ifdef TEST_NODE_LIST
		bool isHit = _hit(&snodes[0], r, t_min, t_max, rec);
		return isHit;
#else
		bool isHit = hit(m_root, r, t_min, t_max, rec);
		return isHit;
#endif
	}

	bool bvh::hit(
		const bvhnode* root,
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec)
	{
		// NOTE
		// https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-ii-tree-traversal-gpu/

		// TODO
		// stack size.
		static const uint32_t stacksize = 64;
		bvhnode* stackbuf[stacksize] = { nullptr };
		bvhnode** stack = &stackbuf[0];

		// push terminator.
		*stack++ = nullptr;

		int stackpos = 1;

		auto node = root;

		do {
			auto left = node->m_left;
			auto right = node->m_right;

#if 0
			bool isHitLeft = false;
			bool isHitRight = false;

			if (left) {
				isHitLeft = left->getBoundingbox().hit(r, t_min, t_max);
			}
			if (right) {
				isHitRight = right->getBoundingbox().hit(r, t_min, t_max);
			}

			if (isHitLeft && left->isLeaf()) {
				hitrecord recLeft;
				if (left->hit(r, t_min, t_max, recLeft)) {
					if (recLeft.t < rec.t) {
						rec = recLeft;
					}
				}
			}
			if (isHitRight && right->isLeaf()) {
				hitrecord recRight;
				if (right->hit(r, t_min, t_max, recRight)) {
					if (recRight.t < rec.t) {
						rec = recRight;
					}
				}
			}

			bool traverseLeft = isHitLeft && !left->isLeaf();
			bool traverseRight = isHitRight && !right->isLeaf();
#else
			bool traverseLeft = false;
			bool traverseRight = false;

			if (left) {
				if (left->isLeaf()) {
					hitrecord recLeft;
					if (left->hit(r, t_min, t_max, recLeft)) {
						if (recLeft.t < rec.t) {
							rec = recLeft;
						}
					}
				}
				else {
					traverseLeft = left->getBoundingbox().hit(r, t_min, t_max);
				}
			}
			if (right) {
				if (right->isLeaf()) {
					hitrecord recRight;
					if (right->hit(r, t_min, t_max, recRight)) {
						if (recRight.t < rec.t) {
							rec = recRight;
						}
					}
				}
				else {
					traverseRight = right->getBoundingbox().hit(r, t_min, t_max);
				}
			}
#endif

			if (!traverseLeft && !traverseRight) {
				node = *(--stack);
				stackpos -= 1;
			}
			else {
				node = traverseLeft ? left : right;

				if (traverseLeft && traverseRight) {
					*(stack++) = right;
					stackpos += 1;
				}
			}
			AT_ASSERT(0 <= stackpos && stackpos < stacksize);

			if (stackpos >= stacksize) {
				return false;
			}
		} while (node != nullptr);

		return (rec.obj != nullptr);
	}

	template<typename T>
	static void pop_front(std::vector<T>& vec)
	{
		AT_ASSERT(!vec.empty());
		vec.erase(vec.begin());
	}

	void bvh::buildBySAH(
		bvhnode* root,
		bvhnode** list,
		uint32_t num)
	{
		// NOTE
		// http://qiita.com/omochi64/items/9336f57118ba918f82ec

#if 0
		// ある？
		AT_ASSERT(num > 0);

		// 全体を覆うAABBを計算.
		root->m_aabb = list[0]->getBoundingbox();
		for (uint32_t i = 1; i < num; i++) {
			auto bbox = list[i]->getBoundingbox();
			root->m_aabb = aabb::merge(root->m_aabb, bbox);
		}

		if (num == 1) {
			// １個しかないので、これだけで終了.
			root->m_left = list[0];
			return;
		}
		else if (num == 2) {
			// ２個だけのときは適当にソートして、終了.
			int axis = (int)(::rand() % 3);

			sortList(list, num, axis);

			root->m_left = list[0];
			root->m_right = list[1];

			return;
		}

		// Triangleとrayのヒットにかかる処理時間の見積もり.
		static const real T_tri = 1;  // 適当.

		// AABBとrayのヒットにかかる処理時間の見積もり.
		static const real T_aabb = 1;  // 適当.

		// 領域分割をせず、polygons を含む葉ノードを構築する場合を暫定の bestCost にする.
		//auto bestCost = T_tri * num;
		uint32_t bestCost = UINT32_MAX;	// 限界まで分割したいので、適当に大きい値にしておく.

		// 分割に最も良い軸 (0:x, 1:y, 2:z)
		int bestAxis = -1;

		// 最も良い分割場所
		int bestSplitIndex = -1;

		// ノード全体のAABBの表面積
		auto rootSurfaceArea = root->m_aabb.computeSurfaceArea();

		for (int axis = 0; axis < 3; axis++) {
			// ポリゴンリストを、それぞれのAABBの中心座標を使い、axis でソートする.
			sortList(list, num, axis);

			// AABBの表面積リスト。s1SA[i], s2SA[i] は、
			// 「S1側にi個、S2側に(polygons.size()-i)個ポリゴンがあるように分割」したときの表面積
			std::vector<real> s1SurfaceArea(num + 1, AT_MATH_INF);
			std::vector<real> s2SurfaceArea(num + 1, AT_MATH_INF);

			// 分割された2つの領域.
			std::vector<bvhnode*> s1;					// 右側.
			std::vector<bvhnode*> s2(list, list + num);	// 左側.

			// NOTE
			// s2側から取り出して、s1に格納するため、s2にリストを全部入れる.

			aabb s1bbox;

			// 可能な分割方法について、s1側の AABB の表面積を計算.
			for (uint32_t i = 0; i <= num; i++) {
				s1SurfaceArea[i] = s1bbox.computeSurfaceArea();

				if (s2.size() > 0) {
					// s2側で、axis について最左 (最小位置) にいるポリゴンをs1の最右 (最大位置) に移す
					auto p = s2.front();
					s1.push_back(p);
					pop_front(s2);

					// 移したポリゴンのAABBをマージしてs1のAABBとする.
					auto bbox = p->getBoundingbox();
					s1bbox = aabb::merge(s1bbox, bbox);
				}
			}

			// 逆にs2側のAABBの表面積を計算しつつ、SAH を計算.
			aabb s2bbox;

			for (int i = num; i >= 0; i--) {
				s2SurfaceArea[i] = s2bbox.computeSurfaceArea();

				if (s1.size() > 0 && s2.size() > 0) {
					// SAH-based cost の計算.
					auto cost =	2 * T_aabb
						+ (s1SurfaceArea[i] * s1.size() + s2SurfaceArea[i] * s2.size()) * T_tri / rootSurfaceArea;

					// 最良コストが更新されたか.
					if (cost < bestCost) {
						bestCost = cost;
						bestAxis = axis;
						bestSplitIndex = i;
					}
				}

				if (s1.size() > 0) {
					// s1側で、axis について最右にいるポリゴンをs2の最左に移す.
					auto p = s1.back();
					
					// 先頭に挿入.
					s2.insert(s2.begin(), p); 

					s1.pop_back();

					// 移したポリゴンのAABBをマージしてS2のAABBとする.
					auto bbox = p->getBoundingbox();
					s2bbox = aabb::merge(s2bbox, bbox);
				}
			}
		}

		if (bestAxis >= 0) {
			// bestAxis に基づき、左右に分割.
			// bestAxis でソート.
			sortList(list, num, bestAxis);

			// 左右の子ノードを作成.
			root->m_left = new bvhnode();
			root->m_right = new bvhnode();

			// リストを分割.
			int leftListNum = bestSplitIndex;
			int rightListNum = num - leftListNum;

			AT_ASSERT(rightListNum > 0);

			// 再帰処理
			buildBySAH(root->m_left, list, leftListNum);
			buildBySAH(root->m_right, list + leftListNum, rightListNum);
		}
#else
		struct BuildInfo {
			bvhnode* node{ nullptr };
			bvhnode** list{ nullptr };
			uint32_t num{ 0 };

			BuildInfo() {}
			BuildInfo(bvhnode* _node, bvhnode** _list, uint32_t _num)
			{
				node = _node;
				list = _list;
				num = _num;
			}
		};

		std::vector<BuildInfo> stacks;
		stacks.push_back(BuildInfo());	// Add terminator.

		BuildInfo info(root, list, num);

		while (info.node != nullptr)
		{
			// 全体を覆うAABBを計算.
			info.node->m_aabb = info.list[0]->getBoundingbox();
			for (uint32_t i = 1; i < info.num; i++) {
				auto bbox = info.list[i]->getBoundingbox();
				info.node->m_aabb = aabb::merge(info.node->m_aabb, bbox);
			}

			if (info.num == 1) {
				// １個しかないので、これだけで終了.
				info.node->m_left = info.list[0];

				info = stacks.back();
				stacks.pop_back();
				continue;
			}
			else if (info.num == 2) {
				// ２個だけのときは適当にソートして、終了.
				int axis = (int)(::rand() % 3);

				sortList(info.list, info.num, axis);

				info.node->m_left = info.list[0];
				info.node->m_right = info.list[1];

				info = stacks.back();
				stacks.pop_back();
				continue;
			}

			// Triangleとrayのヒットにかかる処理時間の見積もり.
			static const real T_tri = 1;  // 適当.

			// AABBとrayのヒットにかかる処理時間の見積もり.
			static const real T_aabb = 1;  // 適当.

			// 領域分割をせず、polygons を含む葉ノードを構築する場合を暫定の bestCost にする.
			//auto bestCost = T_tri * num;
			uint32_t bestCost = UINT32_MAX;	// 限界まで分割したいので、適当に大きい値にしておく.

			// 分割に最も良い軸 (0:x, 1:y, 2:z)
			int bestAxis = -1;

			// 最も良い分割場所
			int bestSplitIndex = -1;

			// ノード全体のAABBの表面積
			auto rootSurfaceArea = info.node->m_aabb.computeSurfaceArea();

			for (int axis = 0; axis < 3; axis++) {
				// ポリゴンリストを、それぞれのAABBの中心座標を使い、axis でソートする.
				sortList(info.list, info.num, axis);

				// AABBの表面積リスト。s1SA[i], s2SA[i] は、
				// 「S1側にi個、S2側に(polygons.size()-i)個ポリゴンがあるように分割」したときの表面積
				std::vector<real> s1SurfaceArea(info.num + 1, AT_MATH_INF);
				std::vector<real> s2SurfaceArea(info.num + 1, AT_MATH_INF);

				// 分割された2つの領域.
				std::vector<bvhnode*> s1;					// 右側.
				std::vector<bvhnode*> s2(info.list, info.list + info.num);	// 左側.

				// NOTE
				// s2側から取り出して、s1に格納するため、s2にリストを全部入れる.

				aabb s1bbox;

				// 可能な分割方法について、s1側の AABB の表面積を計算.
				for (uint32_t i = 0; i <= info.num; i++) {
					s1SurfaceArea[i] = s1bbox.computeSurfaceArea();

					if (s2.size() > 0) {
						// s2側で、axis について最左 (最小位置) にいるポリゴンをs1の最右 (最大位置) に移す
						auto p = s2.front();
						s1.push_back(p);
						pop_front(s2);

						// 移したポリゴンのAABBをマージしてs1のAABBとする.
						auto bbox = p->getBoundingbox();
						s1bbox = aabb::merge(s1bbox, bbox);
					}
				}

				// 逆にs2側のAABBの表面積を計算しつつ、SAH を計算.
				aabb s2bbox;

				for (int i = info.num; i >= 0; i--) {
					s2SurfaceArea[i] = s2bbox.computeSurfaceArea();

					if (s1.size() > 0 && s2.size() > 0) {
						// SAH-based cost の計算.
						auto cost = 2 * T_aabb
							+ (s1SurfaceArea[i] * s1.size() + s2SurfaceArea[i] * s2.size()) * T_tri / rootSurfaceArea;

						// 最良コストが更新されたか.
						if (cost < bestCost) {
							bestCost = cost;
							bestAxis = axis;
							bestSplitIndex = i;
						}
					}

					if (s1.size() > 0) {
						// s1側で、axis について最右にいるポリゴンをs2の最左に移す.
						auto p = s1.back();

						// 先頭に挿入.
						s2.insert(s2.begin(), p);

						s1.pop_back();

						// 移したポリゴンのAABBをマージしてS2のAABBとする.
						auto bbox = p->getBoundingbox();
						s2bbox = aabb::merge(s2bbox, bbox);
					}
				}
			}

			if (bestAxis >= 0) {
				// bestAxis に基づき、左右に分割.
				// bestAxis でソート.
				sortList(info.list, info.num, bestAxis);

				// 左右の子ノードを作成.
				info.node->m_left = new bvhnode();
				info.node->m_right = new bvhnode();

				// リストを分割.
				int leftListNum = bestSplitIndex;
				int rightListNum = info.num - leftListNum;

				AT_ASSERT(rightListNum > 0);

				auto _list = info.list;

				auto _left = info.node->m_left;
				auto _right = info.node->m_right;

				info = BuildInfo(_left, _list, leftListNum);
				stacks.push_back(BuildInfo(_right, _list + leftListNum, rightListNum));
			}
			else {
				// TODO
				info = stacks.back();
				stacks.pop_back();
			}
		}
#endif
	}

	void bvh::collectNodes(std::vector<BVHNode>& nodes) const
	{
		int order = setTraverseOrder(m_root, 0);
		collectNodes(m_root, nodes, nullptr);

		std::map<hitable*, int> cache;

		auto shapes = transformable::getShapes();
		for (auto s : shapes) {
			const auto idx = s->m_traverseOrder;
			if (idx >= 0) {
				const auto& param = s->getParam();

				if (param.type == ShapeType::Instance) {
					auto& node = nodes[idx];

					// Search if object which instance has is collected.
					auto obj = s->getHasObject();
					auto found = cache.find(const_cast<hitable*>(obj));

					if (found != cache.end()) {
						// Collected.
						node.nestid = found->second;
					}
					else {
						// Not collected yet.
						node.nestid = nodes.size();

						order = s->collectInternalNodes(nodes, order, nullptr);
					}
				}
			}
		}
	}

	int bvh::setTraverseOrder(bvhnode* root, int curOrder)
	{
		static const uint32_t stacksize = 64;
		bvhnode* stackbuf[stacksize] = { nullptr };
		bvhnode** stack = &stackbuf[0];

		// push terminator.
		*stack++ = nullptr;

		int stackpos = 1;

		bvhnode* pnode = root;

		int order = curOrder;

		while (pnode != nullptr) {
			bvhnode* pleft = pnode->m_left;
			bvhnode* pright = pnode->m_right;

			pnode->m_traverseOrder = order++;

			if (pnode->isLeaf()) {
				pnode = *(--stack);
				stackpos -= 1;
			}
			else {
				pnode = pleft;

				if (pright) {
					*(stack++) = pright;
					stackpos += 1;
				}
			}
		}

		return order;
	}

	void bvh::collectNodes(bvhnode* root, std::vector<BVHNode>& nodes, bvhnode* parent)
	{
		static const uint32_t stacksize = 64;
		bvhnode* stackbuf[stacksize] = { nullptr };
		bvhnode** stack = &stackbuf[0];

		// push terminator.
		*stack++ = nullptr;

		int stackpos = 1;

		bvhnode* pnode = root;

		while (pnode != nullptr) {
			bvhnode* pleft = pnode->m_left;
			bvhnode* pright = pnode->m_right;

			BVHNode node;

			node.bbox = pnode->getBoundingbox();

			if (pnode->isLeaf()) {
				node.shapeid = transformable::findShapeIdxAsHitable(pnode);

				if (node.shapeid < 0) {
					// Not find shape.
					pnode->setBVHNodeParamInCollectNodes(node);
				}
			}
			else {
				node.left = (pleft ? pleft->m_traverseOrder : -1);
				node.right = (pright ? pright->m_traverseOrder : -1);
			}

			if (parent) {
				AT_ASSERT(node.shapeid < 0);
				node.shapeid = transformable::findShapeIdxAsHitable(parent);
			}

			nodes.push_back(node);

			if (pnode->isLeaf()) {
				pnode = *(--stack);
				stackpos -= 1;
			}
			else {
				pnode = pleft;

				if (pright) {
					*(stack++) = pright;
					stackpos += 1;
				}
			}
		}
	}
}
