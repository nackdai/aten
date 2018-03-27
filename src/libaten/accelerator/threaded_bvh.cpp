#include "accelerator/threaded_bvh.h"
#include "accelerator/bvh.h"
#include "geometry/tranformable.h"
#include "geometry/object.h"

#include <random>
#include <vector>
#include <iterator>

//#pragma optimize( "", off)

// Threaded BVH
// http://www.ci.i.u-tokyo.ac.jp/~hachisuka/tdf2015.pdf

namespace aten {
	void ThreadedBVH::build(
		hitable** list,
		uint32_t num,
		aabb* bbox/*= nullptr*/)
	{
		if (m_isNested) {
			m_bvh.build(list, num, bbox);

			setBoundingBox(m_bvh.getBoundingbox());

			std::vector<ThreadedBvhNodeEntry> threadedBvhNodeEntries;

			// Convert to linear list.
			registerBvhNodeToLinearList(
				m_bvh.getRoot(),
				threadedBvhNodeEntries);
			
			std::vector<int> listParentId;
			m_listThreadedBvhNode.resize(1);

			// Register bvh node for gpu.
			registerThreadedBvhNode(
				true,
				threadedBvhNodeEntries,
				m_listThreadedBvhNode[0], 
				listParentId);

			// Set order.
			setOrder(
				threadedBvhNodeEntries,
				listParentId, 
				m_listThreadedBvhNode[0]);
		}
		else {
			m_bvh.build(list, num, bbox);

			setBoundingBox(m_bvh.getBoundingbox());

			// Gather local-world matrix.
			transformable::gatherAllTransformMatrixAndSetMtxIdx(m_mtxs);

			std::vector<accelerator*> nestedBvhList;
			std::map<hitable*, accelerator*> nestedBvhMap;

			std::vector<ThreadedBvhNodeEntry> threadedBvhNodeEntries;

			// Register to linear list to traverse bvhnode easily.
			auto root = m_bvh.getRoot();
			registerBvhNodeToLinearList(
				root, 
				nullptr, 
				nullptr, 
				aten::mat4::Identity, 
				threadedBvhNodeEntries, 
				nestedBvhList, 
				nestedBvhMap);

			// Copy to keep.
			m_nestedBvh = nestedBvhList;

			std::vector<int> listParentId;

			if (m_enableLayer) {
				// NOTE
				// 0 is for top layer. So, need +1.
				m_listThreadedBvhNode.resize(m_nestedBvh.size() + 1);
			}
			else {
				m_listThreadedBvhNode.resize(1);
			}

			// Register bvh node for gpu.
			registerThreadedBvhNode(
				false,
				threadedBvhNodeEntries,
				m_listThreadedBvhNode[0],
				listParentId);

			// Set traverse order for linear bvh.
			setOrder(threadedBvhNodeEntries, listParentId, m_listThreadedBvhNode[0]);

			// Copy nested threaded bvh nodes to top layer tree.
			if (m_enableLayer) {
				for (int i = 0; i < nestedBvhList.size(); i++) {
					auto node = nestedBvhList[i];

					// TODO
					if (node->getAccelType() == AccelType::ThreadedBvh) {
						auto threadedBvh = static_cast<ThreadedBVH*>(node);

						auto& threadedBvhNodes = threadedBvh->m_listThreadedBvhNode[0];

						// NODE
						// m_listThreadedBvhNode[0] is for top layer.
						std::copy(
							threadedBvhNodes.begin(), 
							threadedBvhNodes.end(), 
							std::back_inserter(m_listThreadedBvhNode[i + 1]));
					}
				}
			}

			//dump(m_listThreadedBvhNode[1], "node.txt");
		}
	}

	void ThreadedBVH::dump(std::vector<ThreadedBvhNode>& nodes, const char* path)
	{
		FILE* fp = nullptr;
		fopen_s(&fp, path, "wt");
		if (!fp) {
			AT_ASSERT(false);
			return;
		}

		for (const auto& n : nodes) {
			fprintf(fp, "%d %d %d %d %d %d (%.3f, %.3f, %.3f) (%.3f, %.3f, %.3f)\n", 
				(int)n.hit, (int)n.miss, (int)n.shapeid, (int)n.primid, (int)n.exid, (int)n.meshid,
				n.boxmin.x, n.boxmin.y, n.boxmin.z,
				n.boxmax.x, n.boxmax.y, n.boxmax.z);
		}

		fclose(fp);
	}

	void ThreadedBVH::registerBvhNodeToLinearList(
		bvhnode* root,
		bvhnode* parentNode,
		hitable* nestParent,
		const aten::mat4& mtxL2W,
		std::vector<ThreadedBvhNodeEntry>& listBvhNode,
		std::vector<accelerator*>& listNestedBvh,
		std::map<hitable*, accelerator*>& nestedBvhMap)
	{
		bvh::registerBvhNodeToLinearList<ThreadedBvhNodeEntry>(
			root,
			parentNode,
			nestParent,
			mtxL2W,
			listBvhNode,
			listNestedBvh,
			nestedBvhMap,
			[this](std::vector<ThreadedBvhNodeEntry>& list, bvhnode* node, hitable* obj, const aten::mat4& mtx)
		{
			list.push_back(ThreadedBvhNodeEntry(node, obj, mtx));
		},
			[this](bvhnode* node, int exid, int subExid)
		{
			if (node->isLeaf()) {
				// NOTE
				// 0 はベースツリーなので、+1 する.
				node->setExternalId(exid + 1);

				if (subExid >= 0) {
					node->setSubExternalId(subExid + 1);
				}
			}
		});
	}

	void ThreadedBVH::registerThreadedBvhNode(
		bool isPrimitiveLeaf,
		const std::vector<ThreadedBvhNodeEntry>& threadedBvhNodeEntries,
		std::vector<ThreadedBvhNode>& threadedBvhNodes,
		std::vector<int>& listParentId)
	{
		threadedBvhNodes.reserve(threadedBvhNodeEntries.size());
		listParentId.reserve(threadedBvhNodeEntries.size());

		for (const auto& entry : threadedBvhNodeEntries) {
			auto node = entry.node;
			auto nestParent = entry.nestParent;

			ThreadedBvhNode gpunode;

			// NOTE
			// Differ set hit/miss index.

			auto bbox = node->getBoundingbox();
			bbox = aten::aabb::transform(bbox, entry.mtxL2W);

			auto parent = node->getParent();
			int parentId = parent ? parent->getTraversalOrder() : -1;
			listParentId.push_back(parentId);

			if (node->isLeaf()) {
				hitable* item = node->getItem();

				// 自分自身のIDを取得.
				gpunode.shapeid = (float)transformable::findShapeIdxAsHitable(item);

				// もしなかったら、ネストしているので親のIDを取得.
				if (gpunode.shapeid < 0) {
					if (nestParent) {
						gpunode.shapeid = (float)transformable::findShapeIdxAsHitable(nestParent);
					}
				}

				// インスタンスの実体を取得.
				auto internalObj = item->getHasObject();

				if (internalObj) {
					item = const_cast<hitable*>(internalObj);
				}

				gpunode.meshid = (float)item->geomid();

				if (isPrimitiveLeaf) {
					// Leaves of this tree are primitive.
					gpunode.primid = (float)face::findIdx(item);
					gpunode.exid = -1.0f;
				}
				else {
					auto exid = node->getExternalId();
					auto subexid = node->getSubExternalId();

					gpunode.noExternal = false;
					gpunode.hasLod = (subexid >= 0);
					gpunode.mainExid = exid;
					gpunode.lodExid = (subexid >= 0 ? subexid : 0);
				}
			}

			gpunode.boxmax = aten::vec4(bbox.maxPos(), 0);
			gpunode.boxmin = aten::vec4(bbox.minPos(), 0);

			threadedBvhNodes.push_back(gpunode);
		}
	}

	void ThreadedBVH::setOrder(
		const std::vector<ThreadedBvhNodeEntry>& threadedBvhNodeEntries,
		const std::vector<int>& listParentId,
		std::vector<ThreadedBvhNode>& threadedBvhNodes)
	{
		auto num = threadedBvhNodes.size();

		for (int n = 0; n < num; n++) {
			auto node = threadedBvhNodeEntries[n].node;
			auto& gpunode = threadedBvhNodes[n];

			bvhnode* next = nullptr;
			if (n + 1 < num) {
				next = threadedBvhNodeEntries[n + 1].node;
			}

			if (node->isLeaf()) {
				// Hit/Miss.
				// Always the next node in the array.
				if (next) {
					gpunode.hit = (float)next->getTraversalOrder();
					gpunode.miss = (float)next->getTraversalOrder();
				}
				else {
					gpunode.hit = -1;
					gpunode.miss = -1;
				}
			}
			else {
				// Hit.
				// Always the next node in the array.
				if (next) {
					gpunode.hit = (float)next->getTraversalOrder();
				}
				else {
					gpunode.hit = -1;
				}

				// Miss.

				// Search the parent.
				auto parentId = listParentId[n];
				bvhnode* parent = (parentId >= 0
					? threadedBvhNodeEntries[parentId].node
					: nullptr);

				if (parent) {
					bvhnode* left = bvh::getInternalNode(parent->getLeft());
					bvhnode* right = bvh::getInternalNode(parent->getRight());

					bool isLeft = (left == node);

					if (isLeft) {
						// Internal, left: sibling node.
						auto sibling = right;
						isLeft = (sibling != nullptr);

						if (isLeft) {
							gpunode.miss = (float)sibling->getTraversalOrder();
						}
					}

					bvhnode* curParent = parent;

					if (!isLeft) {
						// Internal, right: parent's sibling node (until it exists) .
						for (;;) {
							// Search the grand parent.
							auto grandParentId = listParentId[curParent->getTraversalOrder()];
							bvhnode* grandParent = (grandParentId >= 0
								? threadedBvhNodeEntries[grandParentId].node
								: nullptr);

							if (grandParent) {
								bvhnode* _left = bvh::getInternalNode(grandParent->getLeft());;
								bvhnode* _right = bvh::getInternalNode(grandParent->getRight());;

								auto sibling = _right;
								if (sibling) {
									if (sibling != curParent) {
										gpunode.miss = (float)sibling->getTraversalOrder();
										break;
									}
								}
							}
							else {
								gpunode.miss = -1;
								break;
							}

							curParent = grandParent;
						}
					}
				}
				else {
					gpunode.miss = -1;
				}
			}
		}
	}

	bool ThreadedBVH::hit(
		const ray& r,
		real t_min, real t_max,
		Intersection& isect) const
	{
		return hit(0, m_listThreadedBvhNode, r, t_min, t_max, isect);
	}

	bool ThreadedBVH::hit(
		int exid,
		const std::vector<std::vector<ThreadedBvhNode>>& listThreadedBvhNode,
		const ray& r,
		real t_min, real t_max,
		Intersection& isect) const
	{
		auto& shapes = transformable::getShapes();
		auto& prims = face::faces();

		real hitt = AT_MATH_INF;

		int nodeid = 0;

		for (;;) {
			const ThreadedBvhNode* node = nullptr;

			if (nodeid >= 0) {
				node = &listThreadedBvhNode[exid][nodeid];
			}

			if (!node) {
				break;
			}

			bool isHit = false;

			if (node->isLeaf()) {
				Intersection isectTmp;

				auto s = node->shapeid >= 0 ? shapes[(int)node->shapeid] : nullptr;

				if (node->exid >= 0) {
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

					//int exid = node->mainExid;
					int exid = *(int*)(&node->exid);
					exid = AT_BVHNODE_MAIN_EXID(exid);

					isHit = hit(
						exid,
						listThreadedBvhNode,
						transformedRay,
						t_min, t_max,
						isectTmp);

					if (isHit) {
						isectTmp.objid = s->id();
					}
				}
				else if (node->primid >= 0) {
					// Hit test for a primitive.
					auto prim = (hitable*)prims[(int)node->primid];
					isHit = prim->hit(r, t_min, t_max, isectTmp);
					if (isHit) {
						// Set dummy to return if ray hit.
						isectTmp.objid = s ? s->id() : 1;
					}
				}
				else {
					// Hit test for a shape.
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
				isHit = aten::aabb::hit(r, node->boxmin, node->boxmax, t_min, t_max);
			}

			if (isHit) {
				nodeid = (int)node->hit;
			}
			else {
				nodeid = (int)node->miss;
			}
		}

		return (isect.objid >= 0);
	}

	void ThreadedBVH::update()
	{
		m_bvh.update();

		setBoundingBox(m_bvh.getBoundingbox());

		// TODO
		// More efficient. ex) Gather only transformed object etc...
		// Gather local-world matrix.
		m_mtxs.clear();
		transformable::gatherAllTransformMatrixAndSetMtxIdx(m_mtxs);

		auto root = m_bvh.getRoot();
		std::vector<ThreadedBvhNodeEntry> threadedBvhNodeEntries;
		registerBvhNodeToLinearList(root, threadedBvhNodeEntries);

		std::vector<int> listParentId;
		listParentId.reserve(threadedBvhNodeEntries.size());

		m_listThreadedBvhNode[0].clear();

		for (auto& entry : threadedBvhNodeEntries) {
			auto node = entry.node;

			ThreadedBvhNode gpunode;

			// NOTE
			// Differ set hit/miss index.

			auto bbox = node->getBoundingbox();

			auto parent = node->getParent();
			int parentId = parent ? parent->getTraversalOrder() : -1;
			listParentId.push_back(parentId);

			if (node->isLeaf()) {
				hitable* item = node->getItem();

				// 自分自身のIDを取得.
				gpunode.shapeid = (float)transformable::findShapeIdxAsHitable(item);
				AT_ASSERT(gpunode.shapeid >= 0);

				// インスタンスの実体を取得.
				auto internalObj = item->getHasObject();

				if (internalObj) {
					item = const_cast<hitable*>(internalObj);
				}

				gpunode.meshid = (float)item->geomid();

				int exid = node->getExternalId();
				int subexid = node->getSubExternalId();

				if (exid < 0) {
					gpunode.exid = -1.0f;
				}
				else {
					gpunode.noExternal = false;
					gpunode.hasLod = (subexid >= 0);
					gpunode.mainExid = (exid >= 0 ? exid : 0);
					gpunode.lodExid = (subexid >= 0 ? subexid : 0);
				}
			}

			gpunode.boxmax = aten::vec4(bbox.maxPos(), 0);
			gpunode.boxmin = aten::vec4(bbox.minPos(), 0);

			m_listThreadedBvhNode[0].push_back(gpunode);
		}

		setOrder(threadedBvhNodeEntries, listParentId, m_listThreadedBvhNode[0]);
	}

	void ThreadedBVH::registerBvhNodeToLinearList(
		bvhnode* root, 
		std::vector<ThreadedBvhNodeEntry>& nodes)
	{
		if (!root) {
			return;
		}

		int order = nodes.size();
		root->setTraversalOrder(order);

		nodes.push_back(ThreadedBvhNodeEntry(root, nullptr, aten::mat4::Identity));

		registerBvhNodeToLinearList(root->getLeft(), nodes);
		registerBvhNodeToLinearList(root->getRight(), nodes);
	}
}
