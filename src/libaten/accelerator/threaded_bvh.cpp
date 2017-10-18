#include "accelerator/threaded_bvh.h"
#include "accelerator/bvh.h"
#include "shape/tranformable.h"
#include "object/object.h"

#include <random>
#include <vector>

//#pragma optimize( "", off)

// Threaded BVH
// http://www.ci.i.u-tokyo.ac.jp/~hachisuka/tdf2015.pdf

namespace aten {
	void ThreadedBVH::build(
		hitable** list,
		uint32_t num)
	{
		m_bvh.build(list, num);

		setBoundingBox(m_bvh.getBoundingbox());

		// Gather local-world matrix.
		transformable::gatherAllTransformMatrixAndSetMtxIdx(m_mtxs);

		std::vector<accelerator*> listBvh;
		std::map<hitable*, std::vector<accelerator*>> nestedBvhMap;

		std::vector<std::vector<ThreadedBvhNodeEntry>> listBvhNode;
		
		// Register to linear list to traverse bvhnode easily.
		auto root = m_bvh.getRoot();
		listBvhNode.push_back(std::vector<ThreadedBvhNodeEntry>());
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

			listBvhNode.push_back(std::vector<ThreadedBvhNodeEntry>());
			std::vector<accelerator*> dummy;

			registerBvhNodeToLinearList(root, nullptr, parent, aten::mat4::Identity, listBvhNode[i + 1], dummy, nestedBvhMap);
			AT_ASSERT(dummy.empty());
		}

		// Register bvh node for gpu.
		for (int i = 0; i < listBvhNode.size(); i++) {
			m_listGpuBvhNode.push_back(std::vector<GPUBvhNode>());

			// Leaves of nested bvh are primitive.
			// Index 0 is primiary tree, and Index N (N > 0) is nested tree.
			bool isPrimitiveLeaf = (i > 0);

			registerGpuBvhNode(isPrimitiveLeaf, listBvhNode[i], m_listGpuBvhNode[i]);
		}

		// Set traverse order for linear bvh.
		for (int i = 0; i < listBvhNode.size(); i++) {
			setOrderForLinearBVH(listBvhNode[i], m_listGpuBvhNode[i]);
		}

		//dump(m_listGpuBvhNode[1], "node.txt");
	}

	void ThreadedBVH::dump(std::vector<GPUBvhNode>& nodes, const char* path)
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
		std::vector<accelerator*>& listBvh,
		std::map<hitable*, std::vector<accelerator*>>& nestedBvhMap)
	{
		bvh::registerBvhNodeToLinearList<ThreadedBvhNodeEntry>(
			root,
			parentNode,
			nestParent,
			mtxL2W,
			listBvhNode,
			listBvh,
			nestedBvhMap,
			[this](std::vector<ThreadedBvhNodeEntry>& list, bvhnode* node, hitable* obj, const aten::mat4& mtx)
		{
			list.push_back(ThreadedBvhNodeEntry(node, obj, mtx));
		},
			[this](bvhnode* node)
		{
			if (node->isLeaf()) {
				node->setExternalId(m_exid);
				m_exid++;
			}
		});
	}

	void ThreadedBVH::registerGpuBvhNode(
		bool isPrimitiveLeaf,
		std::vector<ThreadedBvhNodeEntry>& listBvhNode,
		std::vector<GPUBvhNode>& listGpuBvhNode)
	{
		listGpuBvhNode.reserve(listBvhNode.size());

		for (const auto& entry : listBvhNode) {
			auto node = entry.node;
			auto nestParent = entry.nestParent;

			GPUBvhNode gpunode;

			// NOTE
			// Differ set hit/miss index.

			auto bbox = node->getBoundingbox();
			bbox = aten::aabb::transform(bbox, entry.mtxL2W);

			auto parent = node->getParent();
			gpunode.parent = (float)(parent ? parent->getTraversalOrder() : -1);

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

				gpunode.meshid = (float)item->meshid();

				if (isPrimitiveLeaf) {
					// Leaves of this tree are primitive.
					gpunode.primid = (float)face::findIdx(item);
					gpunode.exid = -1.0f;
				}
				else {
					gpunode.exid = (float)node->getExternalId();
				}
			}

			gpunode.boxmax = aten::vec4(bbox.maxPos(), 0);
			gpunode.boxmin = aten::vec4(bbox.minPos(), 0);

			listGpuBvhNode.push_back(gpunode);
		}
	}

	void ThreadedBVH::setOrderForLinearBVH(
		std::vector<ThreadedBvhNodeEntry>& listBvhNode,
		std::vector<GPUBvhNode>& listGpuBvhNode)
	{
		auto num = listGpuBvhNode.size();

		for (int n = 0; n < num; n++) {
			auto node = listBvhNode[n].node;
			auto& gpunode = listGpuBvhNode[n];

			bvhnode* next = nullptr;
			if (n + 1 < num) {
				next = listBvhNode[n + 1].node;
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
				bvhnode* parent = (gpunode.parent >= 0.0f
					? listBvhNode[(int)gpunode.parent].node
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
							const auto& parentNode = listGpuBvhNode[curParent->getTraversalOrder()];
							bvhnode* grandParent = (parentNode.parent >= 0
								? listBvhNode[(int)parentNode.parent].node
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
		return hit(0, m_listGpuBvhNode, r, t_min, t_max, isect);
	}

	bool ThreadedBVH::hit(
		int exid,
		const std::vector<std::vector<GPUBvhNode>>& listGpuBvhNode,
		const ray& r,
		real t_min, real t_max,
		Intersection& isect) const
	{
		static const uint32_t stacksize = 64;

		auto& shapes = transformable::getShapes();
		auto& prims = face::faces();

		real hitt = AT_MATH_INF;

		int nodeid = 0;

		for (;;) {
			const GPUBvhNode* node = nullptr;

			if (nodeid >= 0) {
				node = &listGpuBvhNode[exid][nodeid];
			}

			if (!node) {
				break;
			}

			bool isHit = false;

			if (node->isLeaf()) {
				Intersection isectTmp;

				auto s = shapes[(int)node->shapeid];

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

					isHit = hit(
						(int)node->exid,
						listGpuBvhNode,
						transformedRay,
						t_min, t_max,
						isectTmp);
				}
				else if (node->primid >= 0) {
					// Hit test for a primitive.
					auto prim = (hitable*)prims[(int)node->primid];
					isHit = prim->hit(r, t_min, t_max, isectTmp);
					if (isHit) {
						isectTmp.objid = s->id();
					}
				}
				else {
					// Hit test for a shape.
					isHit = s->hit(r, t_min, t_max, isectTmp);
				}

				if (isHit) {
					if (isectTmp.t < isect.t) {
						isect = isectTmp;
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
}
