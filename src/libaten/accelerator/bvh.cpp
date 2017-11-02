#include "accelerator/bvh.h"
#include "shape/tranformable.h"
#include "object/object.h"

#include <random>
#include <vector>

//#define TEST_NODE_LIST
//#pragma optimize( "", off)

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

	void sortList(hitable**& list, uint32_t num, int axis)
	{
		switch (axis) {
		case 0:
			::qsort(list, num, sizeof(hitable*), compareX);
			break;
		case 1:
			::qsort(list, num, sizeof(hitable*), compareY);
			break;
		default:
			::qsort(list, num, sizeof(hitable*), compareZ);
			break;
		}
	}

	bool bvhnode::hit(
		const ray& r,
		real t_min, real t_max,
		Intersection& isect) const
	{
#if 0
		if (m_item) {
			return m_item->hit(r, t_min, t_max, isect);
		}
#else
		if (m_childrenNum > 0) {
			bool isHit = false;

			for (int i = 0; i < m_childrenNum; i++) {
				Intersection isectTmp;
				isectTmp.t = AT_MATH_INF;
				auto res = m_children[i]->hit(r, t_min, t_max, isectTmp);
				
				if (res) {
					if (isectTmp.t < isect.t) {
						isect = isectTmp;
						t_max = isect.t;

						isHit = true;
					}
				}
			}

			return isHit;
		}
		else if (m_item) {
			return m_item->hit(r, t_min, t_max, isect);
		}
#endif
		else {
			auto bbox = getBoundingbox();
			auto isHit = bbox.hit(r, t_min, t_max);

			if (isHit) {
				isHit = bvh::hit(this, r, t_min, t_max, isect);
			}

			return isHit;
		}
	}

	///////////////////////////////////////////////////////

	void bvh::build(
		hitable** list,
		uint32_t num)
	{
		// TODO
		//int axis = (int)(::rand() % 3);
		int axis = 0;

		sortList(list, num, axis);

		m_root = new bvhnode(nullptr);
		buildBySAH(m_root, list, num);
	}

	bool bvh::hit(
		const ray& r,
		real t_min, real t_max,
		Intersection& isect) const
	{
		bool isHit = hit(m_root, r, t_min, t_max, isect);
		return isHit;
	}

	bool bvh::hit(
		const bvhnode* root,
		const ray& r,
		real t_min, real t_max,
		Intersection& isect)
	{
		// NOTE
		// https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-ii-tree-traversal-gpu/

		// TODO
		// stack size.
		static const uint32_t stacksize = 64;
		const bvhnode* stackbuf[stacksize];

		stackbuf[0] = root;
		int stackpos = 1;

		while (stackpos > 0) {
			auto node = stackbuf[stackpos - 1];

			stackpos -= 1;

			if (node->isLeaf()) {
				Intersection isectTmp;
				if (node->hit(r, t_min, t_max, isectTmp)) {
					if (isectTmp.t < isect.t) {
						isect = isectTmp;
						t_max = isect.t;
					}
				}
			}
			else {
				if (node->getBoundingbox().hit(r, t_min, t_max)) {
					if (node->m_left) {
						stackbuf[stackpos++] = node->m_left;
					}
					if (node->m_right) {
						stackbuf[stackpos++] = node->m_right;
					}

					if (stackpos > stacksize) {
						AT_ASSERT(false);
						return false;
					}
				}
			}
		}

		return (isect.objid >= 0);
	}

	template<typename T>
	static void pop_front(std::vector<T>& vec)
	{
		AT_ASSERT(!vec.empty());
		vec.erase(vec.begin());
	}

	void bvh::buildBySAH(
		bvhnode* root,
		hitable** list,
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
			hitable** list{ nullptr };
			uint32_t num{ 0 };

			BuildInfo() {}
			BuildInfo(bvhnode* _node, hitable** _list, uint32_t _num)
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
			info.node->setBoundingBox(info.list[0]->getBoundingbox());
			for (uint32_t i = 1; i < info.num; i++) {
				auto bbox = info.list[i]->getBoundingbox();
				info.node->setBoundingBox(
					aabb::merge(info.node->getBoundingbox(), bbox));
			}

#ifdef ENABLE_BVH_MULTI_TRIANGLES
			if (info.num <= 4) {
				bool canRegisterAsChild = true;

				if (info.num > 1) {
					for (int i = 0; i < info.num; i++) {
						auto internalObj = info.list[i]->getHasObject();
						if (internalObj) {
							canRegisterAsChild = false;
							break;
						}
					}
				}

				if (canRegisterAsChild) {
					// TODO
					int axis = 0;

					sortList(info.list, info.num, axis);

					for (int i = 0; i < info.num; i++) {
						info.node->registerChild(info.list[i], i);
					}

					info.node->setChildrenNum(info.num);

					info = stacks.back();
					stacks.pop_back();
					continue;
		}
			}
#else
			if (info.num == 1) {
				// １個しかないので、これだけで終了.
				info.node->m_left = new bvhnode(info.node, info.list[0]);

				info.node->m_left->setBoundingBox(info.list[0]->getBoundingbox());

				info = stacks.back();
				stacks.pop_back();
				continue;
			}
			else if (info.num == 2) {
				// ２個だけのときは適当にソートして、終了.

				// TODO
				//int axis = (int)(::rand() % 3);
				int axis = 0;

				sortList(info.list, info.num, axis);

				info.node->m_left = new bvhnode(info.node, info.list[0]);
				info.node->m_right = new bvhnode(info.node, info.list[1]);

				info.node->m_left->setBoundingBox(info.list[0]->getBoundingbox());
				info.node->m_right->setBoundingBox(info.list[1]->getBoundingbox());

				info = stacks.back();
				stacks.pop_back();
				continue;
			}			
#endif

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
			auto rootSurfaceArea = info.node->getBoundingbox().computeSurfaceArea();

			for (int axis = 0; axis < 3; axis++) {
				// ポリゴンリストを、それぞれのAABBの中心座標を使い、axis でソートする.
				sortList(info.list, info.num, axis);

				// AABBの表面積リスト。s1SA[i], s2SA[i] は、
				// 「S1側にi個、S2側に(polygons.size()-i)個ポリゴンがあるように分割」したときの表面積
				std::vector<real> s1SurfaceArea(info.num + 1, AT_MATH_INF);
				std::vector<real> s2SurfaceArea(info.num + 1, AT_MATH_INF);

				// 分割された2つの領域.
				std::vector<hitable*> s1;					// 右側.
				std::vector<hitable*> s2(info.list, info.list + info.num);	// 左側.

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
				info.node->m_left = new bvhnode(info.node);
				info.node->m_right = new bvhnode(info.node);

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

	bvhnode* bvh::getInternalNode(bvhnode* node, aten::mat4* mtxL2W/*= nullptr*/)
	{
		bvhnode* ret = nullptr;

		if (node) {
			auto item = node->getItem();
			if (item) {
				auto internalObj = const_cast<hitable*>(item->getHasObject());
				if (internalObj) {
					auto t = transformable::getShapeAsHitable(item);

					if (mtxL2W) {
						aten::mat4 mtxW2L;
						t->getMatrices(*mtxL2W, mtxW2L);
					}

					auto accel = internalObj->getInternalAccelerator();

					// NOTE
					// 本来ならこのキャストは不正だが、BVHであることは自明なので.
					auto bvh = *(aten::bvh**)&accel;

					ret = bvh->getRoot();
				}
			}

			if (!ret) {
				ret = node;
			}
		}

		return ret;
	}

	accelerator::ResultIntersectTestByFrustum bvh::intersectTestByFrustum(const frustum& f)
	{
		std::stack<Candidate> stack;

		accelerator::ResultIntersectTestByFrustum result;

		findCandidates(m_root, nullptr, f, stack);

		bvhnode* candidate = nullptr;

		while (!stack.empty()) {
			auto c = stack.top();
			stack.pop();

			auto instanceNode = c.instanceNode;

			bvhnode* node = c.node;
			int exid = 0;

			aten::mat4 mtxW2L;

			if (instanceNode) {
				node = getInternalNode(node, &mtxW2L);

				aten::hitable* originalItem = node->getItem();
				AT_ASSERT(originalItem->isInstance());

				// Register relation between instance and nested bvh.
				auto internalItem = const_cast<hitable*>(originalItem->getHasObject());

				// TODO
				auto obj = (AT_NAME::object*)internalItem;

				auto nestedBvh = (bvh*)obj->getInternalAccelerator();

				node = nestedBvh->m_root;
				mtxW2L.invert();

				exid = instanceNode->getExternalId();
			}

			candidate = node;

			auto transformedFrustum = f;
			transformedFrustum.transform(mtxW2L);

			auto n = traverse(node, transformedFrustum);

			if (n) {
				candidate = n;
				
				result.ep = candidate->getTraversalOrder();
				result.ex = exid;
				result.top = instanceNode->getTraversalOrder();

				break;
			}
		}

		return std::move(result);
	}

	bvhnode* bvh::traverse(
		bvhnode* root,
		const frustum& f)
	{
		bvhnode* stack[32];
		int stackpos = 0;

		stack[0] = root;
		stackpos = 1;

		while (stackpos > 0) {
			bvhnode* node = stack[stackpos - 1];
			stackpos--;

			if (node->isLeaf()) {
				auto i = f.intersect(node->getBoundingbox());

				if (i != frustum::Intersect::Miss) {
					return node;
				}
			}
			else {
				auto i = f.intersect(node->getBoundingbox());

				if (i != frustum::Intersect::Miss) {
					auto left = node->getLeft();
					auto right = node->getRight();

					if (left && !left->isCandidate()) {
						stack[stackpos++] = left;
					}
					if (right && !right->isCandidate()) {
						stack[stackpos++] = right;
					}
				}
			}
		}

		return nullptr;
	}

	bool bvh::findCandidates(
		bvhnode* node,
		bvhnode* instanceNode,
		const frustum& f,
		std::stack<Candidate>& stack)
	{
		if (!node) {
			return false;
		}

		node->setIsCandidate(false);

		if (node->isLeaf()) {
			auto original = node;
			bvh* nestedBvh = nullptr;
			aten::mat4 mtxW2L;

			// Check if node has nested bvh.
			{
				node = getInternalNode(original, &mtxW2L);

				if (node != original) {
					// Register nested bvh.
					aten::hitable* originalItem = original->getItem();
					AT_ASSERT(originalItem->isInstance());

					// Register relation between instance and nested bvh.
					auto internalItem = const_cast<hitable*>(originalItem->getHasObject());

					// TODO
					auto obj = (AT_NAME::object*)internalItem;

					nestedBvh = (bvh*)obj->getInternalAccelerator();

					mtxW2L.invert();
				}

				node = original;
			}

			auto i = f.intersect(node->getBoundingbox());

			if (i != frustum::Intersect::Miss) {
				if (nestedBvh) {
					// Node has nested bvh.
					auto transformedFrustum = f;
					transformedFrustum.transform(mtxW2L);

					bool found = findCandidates(
						nestedBvh->m_root,
						original,
						transformedFrustum,
						stack);
				}
				else {
					node->setIsCandidate(true);

					stack.push(Candidate(node, instanceNode));
				}

				return true;
			}
		}

		auto s0 = findCandidates(node->getLeft(), instanceNode, f, stack);
		auto s1 = findCandidates(node->getRight(), instanceNode, f, stack);

		if (s0 || s1) {
			node->setIsCandidate(true);
			stack.push(Candidate(node, instanceNode));
		}

		return s0 || s1;
	}
}
