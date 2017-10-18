#include "accelerator/qbvh.h"

//#pragma optimize( "", off)

namespace aten
{
	void qbvh::build(
		hitable** list,
		uint32_t num)
	{
		m_bvh.build(list, num);

		setBoundingBox(m_bvh.getBoundingbox());

		// Gather local-world matrix.
		transformable::gatherAllTransformMatrixAndSetMtxIdx(m_mtxs);

		std::vector<accelerator*> listBvh;
		std::map<hitable*, std::vector<accelerator*>> nestedBvhMap;

		std::vector<std::vector<BvhNode>> listBvhNode;

		// Register to linear list to traverse bvhnode easily.
		auto root = m_bvh.getRoot();
		listBvhNode.push_back(std::vector<BvhNode>());
		registerBvhNodeToLinearList(root, nullptr, nullptr, aten::mat4::Identity, listBvhNode[0], listBvh, nestedBvhMap);

		for (int i = 0; i < listBvh.size(); i++) {
			// TODO
			auto bvh = (aten::bvh*)listBvh[i];

			root = bvh->getRoot();

			hitable* parent = nullptr;

			// TODO
			// Find parent.
			for (auto it : nestedBvhMap) {
				auto& list = it.second;
				auto found = std::find(list.begin(), list.end(), bvh);

				if (found != list.end()) {
					// Found nested bvh.
					parent = it.first;
					break;
				}
			}

			listBvhNode.push_back(std::vector<BvhNode>());
			std::vector<accelerator*> dummy;

			registerBvhNodeToLinearList(root, nullptr, parent, aten::mat4::Identity, listBvhNode[i + 1], dummy, nestedBvhMap);
			AT_ASSERT(dummy.empty());
		}

		m_listQbvhNode.resize(listBvhNode.size());

		// Convert to QBVH.
		for (int i = 0; i < listBvhNode.size(); i++) {
			bool isPrimitiveLeafBvh = (i > 0);

			auto numNodes = convertFromBvh(
				isPrimitiveLeafBvh,
				listBvhNode[i],
				m_listQbvhNode[i]);
		}
	}

	void qbvh::registerBvhNodeToLinearList(
		bvhnode* root,
		bvhnode* parentNode,
		hitable* nestParent,
		const aten::mat4& mtxL2W,
		std::vector<BvhNode>& listBvhNode,
		std::vector<accelerator*>& listBvh,
		std::map<hitable*, std::vector<accelerator*>>& nestedBvhMap)
	{
		bvh::registerBvhNodeToLinearList<BvhNode>(
			root,
			parentNode,
			nestParent,
			mtxL2W,
			listBvhNode,
			listBvh,
			nestedBvhMap,
			[this](std::vector<BvhNode>& list, bvhnode* node, hitable* obj, const aten::mat4& mtx)
		{
			list.push_back(BvhNode(node, obj, mtx));
		},
			[this](bvhnode* node)
		{
			if (node->isLeaf()) {
				node->setExternalId(m_exid);
				m_exid++;
			}
		});
	}

	uint32_t qbvh::convertFromBvh(
		bool isPrimitiveLeaf,
		std::vector<BvhNode>& listBvhNode,
		std::vector<QbvhNode>& listQbvhNode)
	{
		struct QbvhStackEntry {
			uint32_t qbvhNodeIdx;
			uint32_t bvhNodeIdx;

			QbvhStackEntry(uint32_t qbvh, uint32_t bvh)
				: qbvhNodeIdx(qbvh), bvhNodeIdx(bvh)
			{}
			QbvhStackEntry() {}
		};

		listQbvhNode.reserve(listBvhNode.size());
		listQbvhNode.push_back(QbvhNode());

		QbvhStackEntry stack[256];
		stack[0] = QbvhStackEntry(0, 0);

		int stackPos = 1;
		uint32_t numNodes = 1;

		int children[4];

		while (stackPos > 0) {
			auto top = stack[--stackPos];

			auto& qbvhNode = listQbvhNode[top.qbvhNodeIdx];
			const auto& bvhNode = listBvhNode[top.bvhNodeIdx];

			int numChildren = getChildren(listBvhNode, top.bvhNodeIdx, children);

			if (numChildren == 0) {
				// No children, so it is a leaf.
				setQbvhNodeLeafParams(isPrimitiveLeaf, bvhNode, qbvhNode);
				continue;
			}

			// Fill node.
			fillQbvhNode(
				qbvhNode,
				listBvhNode,
				children,
				numChildren);

			qbvhNode.leftChildrenIdx = numNodes;
			qbvhNode.numChildren = numChildren;

			// push all children to the stack
			for (int i = 0; i < numChildren; i++) {
				stack[stackPos++] = QbvhStackEntry(numNodes, children[i]);

				listQbvhNode.push_back(QbvhNode());
				++numNodes;
			}
		}

		return numNodes;
	}

	void qbvh::setQbvhNodeLeafParams(
		bool isPrimitiveLeaf,
		const BvhNode& bvhNode,
		QbvhNode& qbvhNode)
	{
		auto node = bvhNode.node;
		auto nestParent = bvhNode.nestParent;

		auto bbox = node->getBoundingbox();
		bbox = aten::aabb::transform(bbox, bvhNode.mtxL2W);

#if 0
		auto parent = node->getParent();
		qbvhNode.parent = (float)(parent ? parent->getTraversalOrder() : -1);
#endif

		qbvhNode.leftChildrenIdx = 0;

		qbvhNode.bmaxx.set(real(0));
		qbvhNode.bmaxy.set(real(0));
		qbvhNode.bmaxz.set(real(0));

		qbvhNode.bminx.set(real(0));
		qbvhNode.bminy.set(real(0));
		qbvhNode.bminz.set(real(0));

		qbvhNode.numChildren = 0;

		if (node->isLeaf()) {
			hitable* item = node->getItem();

			// 自分自身のIDを取得.
			qbvhNode.shapeid = transformable::findShapeIdxAsHitable(item);

			// もしなかったら、ネストしているので親のIDを取得.
			if (qbvhNode.shapeid < 0) {
				if (nestParent) {
					qbvhNode.shapeid = transformable::findShapeIdxAsHitable(nestParent);
				}
			}

			// インスタンスの実体を取得.
			auto internalObj = item->getHasObject();

			if (internalObj) {
				item = const_cast<hitable*>(internalObj);
			}

			qbvhNode.meshid = item->meshid();

			if (isPrimitiveLeaf) {
				// Leaves of this tree are primitive.
				qbvhNode.primid = face::findIdx(item);
				qbvhNode.exid = -1;
			}
			else {
				qbvhNode.exid = node->getExternalId();
			}

			qbvhNode.isLeaf = true;
		}
	}

	void qbvh::fillQbvhNode(
		QbvhNode& qbvhNode,
		std::vector<BvhNode>& listBvhNode,
		int children[4],
		int numChildren)
	{
		for (int i = 0; i < numChildren; i++) {
			int childIdx = children[i];
			const auto& bvhNode = listBvhNode[childIdx];

			const auto node = bvhNode.node;
			
			auto bbox = node->getBoundingbox();
			bbox = aten::aabb::transform(bbox, bvhNode.mtxL2W);

			const auto& bmax = bbox.maxPos();
			const auto& bmin = bbox.minPos();

			qbvhNode.bmaxx[i] = bmax.x;
			qbvhNode.bmaxy[i] = bmax.y;
			qbvhNode.bmaxz[i] = bmax.z;

			qbvhNode.bminx[i] = bmin.x;
			qbvhNode.bminy[i] = bmin.y;
			qbvhNode.bminz[i] = bmin.z;
		}

		// Set 0s for empty child.
		for (int i = numChildren; i < 4; i++) {
			qbvhNode.bmaxx[i] = real(0);
			qbvhNode.bmaxy[i] = real(0);
			qbvhNode.bmaxz[i] = real(0);

			qbvhNode.bminx[i] = real(0);
			qbvhNode.bminy[i] = real(0);
			qbvhNode.bminz[i] = real(0);

			qbvhNode.leftChildrenIdx = 0;
		}

		qbvhNode.isLeaf = false;
	}

	int qbvh::getChildren(
		std::vector<BvhNode>& listBvhNode,
		int bvhNodeIdx,
		int children[4])
	{
		const auto bvhNode = listBvhNode[bvhNodeIdx].node;

		// Invalidate children.
		children[0] = children[1] = children[2] = children[3] = -1;
		int numChildren = 0;

		if (bvhNode->isLeaf()) {
			// No children.
			return numChildren;
		}

		const auto left = bvhNode->getLeft();

		if (left) {
			if (left->isLeaf()) {
				children[numChildren++] = left->getTraversalOrder();
			}
			else {
				const auto left_left = left->getLeft();
				const auto left_right = left->getRight();

				if (left_left) {
					children[numChildren++] = left_left->getTraversalOrder();
				}
				if (left_right) {
					children[numChildren++] = left_right->getTraversalOrder();
				}
			}
		}

		const auto right = bvhNode->getRight();

		if (right) {
			if (right->isLeaf()) {
				children[numChildren++] = right->getTraversalOrder();
			}
			else {
				const auto right_left = right->getLeft();
				const auto right_right = right->getRight();

				if (right_left) {
					children[numChildren++] = right_left->getTraversalOrder();
				}
				if (right_right) {
					children[numChildren++] = right_right->getTraversalOrder();
				}
			}
		}

		return numChildren;
	}

	inline int intersectAABB(
		aten::vec4& result,
		const aten::ray& r,
		real t_min, real t_max,
		const aten::vec4& bminx, const aten::vec4& bmaxx,
		const aten::vec4& bminy, const aten::vec4& bmaxy,
		const aten::vec4& bminz, const aten::vec4& bmaxz)
	{
		// NOTE
		// No SSE...

		aten::vec3 invdir = real(1) / (r.dir + aten::vec3(real(1e-6)));
		aten::vec3 oxinvdir = -r.org * invdir;

		aten::vec4 invdx(invdir.x);
		aten::vec4 invdy(invdir.y);
		aten::vec4 invdz(invdir.z);

		aten::vec4 ox(oxinvdir.x);
		aten::vec4 oy(oxinvdir.y);
		aten::vec4 oz(oxinvdir.z);

		aten::vec4 minus_inf(-AT_MATH_INF);
		aten::vec4 plus_inf(AT_MATH_INF);

		// X 
		auto fx = bmaxx * invdx + ox;
		auto nx = bminx * invdx + ox;

		// Y
		auto fy = bmaxy * invdy + oy;
		auto ny = bminy * invdy + oy;

		// Z
		auto fz = bmaxz * invdz + oz;
		auto nz = bminz * invdz + oz;

		auto tmaxX = max(fx, nx);
		auto tminX = min(fx, nx);

		auto tmaxY = max(fy, ny);
		auto tminY = min(fy, ny);

		auto tmaxZ = max(fz, nz);
		auto tminZ = min(fz, nz);

		auto t1 = min(min(tmaxX, tmaxY), min(tmaxZ, t_max));
		auto t0 = max(max(tminX, tminY), max(tminZ, t_min));

		union isHit {
			struct {
				uint8_t _0 : 1;
				uint8_t _1 : 1;
				uint8_t _2 : 1;
				uint8_t _3 : 1;
				uint8_t padding : 4;
			};
			uint8_t f;
		} hit;

		hit.f = 0;
		hit._0 = (t0.x <= t1.x);
		hit._1 = (t0.y <= t1.y);
		hit._2 = (t0.z <= t1.z);
		hit._3 = (t0.w <= t1.w);

		result = t0;

		return hit.f;
	}

	bool qbvh::hit(
		const ray& r,
		real t_min, real t_max,
		Intersection& isect) const
	{
		return hit(0, m_listQbvhNode, r, t_min, t_max, isect);
	}

	bool qbvh::hit(
		int exid,
		const std::vector<std::vector<QbvhNode>>& listQbvhNode,
		const ray& r,
		real t_min, real t_max,
		Intersection& isect) const
	{
		static const uint32_t stacksize = 64;

		struct Intersect {
			const QbvhNode* node{ nullptr };
			real t;

			Intersect(const QbvhNode* n, real _t) : node(n), t(_t) {}
			Intersect() {}
		} stackbuf[stacksize];

		auto& shapes = transformable::getShapes();
		auto& prims = face::faces();

		stackbuf[0] = Intersect(&listQbvhNode[exid][0], t_max);
		int stackpos = 1;

		while (stackpos > 0) {
			const auto& node = stackbuf[stackpos - 1];
			stackpos -= 1;

			if (node.t > t_max) {
				continue;
			}

			auto pnode = node.node;

			const auto numChildren = pnode->numChildren;

			if (pnode->isLeaf) {
				Intersection isectTmp;

				bool isHit = false;

				auto s = shapes[(int)pnode->shapeid];

				if (pnode->exid >= 0) {
					// Traverse external linear bvh list.
					const auto& param = s->getParam();

					int mtxid = param.mtxid;

					aten::ray transformedRay;

					if (mtxid >= 0) {
						const auto& mtxW2L = m_mtxs[mtxid * 2 + 1];

						transformedRay = mtxW2L.applyRay(r);
					}
					else {
						transformedRay = r;
					}

					isHit = hit(
						(int)pnode->exid,
						listQbvhNode,
						transformedRay,
						t_min, t_max,
						isectTmp);
				}
				else if (pnode->primid >= 0) {
					auto f = prims[pnode->primid];
					isHit = f->hit(r, t_min, t_max, isectTmp);

					if (isHit) {
						isectTmp.objid = s->id();
					}
				}
				else {
					// sphere, cube.
					isHit = s->hit(r, t_min, t_max, isectTmp);
				}

				if (isHit) {
					if (isectTmp.t < isect.t) {
						isect = isectTmp;
						t_max = isect.t;
					}
				}
			}
			else {
				// Hit test children aabb.
				aten::vec4 interserctT;
				auto res = intersectAABB(
					interserctT,
					r,
					t_min, t_max,
					pnode->bminx, pnode->bmaxx,
					pnode->bminy, pnode->bmaxy,
					pnode->bminz, pnode->bmaxz);

				// Stack hit children.
				for (int i = 0; i < numChildren; i++) {
					if ((res & (1 << i)) > 0) {
						stackbuf[stackpos] = Intersect(
							&listQbvhNode[exid][pnode->leftChildrenIdx + i],
							interserctT[i]);
						stackpos++;
					}
				}
			}
		}

		return (isect.objid >= 0);
	}
}
