#include "accelerator/gpu_bvh.h"
#include "accelerator/bvh.h"
#include "shape/tranformable.h"
#include "object/object.h"

#include <random>
#include <vector>

//#pragma optimize( "", off)

namespace aten {
	static std::vector<std::vector<GPUBvhNode>> s_listGpuBvhNode;
	static std::vector<aten::mat4> s_mtxs;

	// TODO
	static std::map<hitable*, std::vector<accelerator*>> s_nestedBvhMap;

	void GPUBvh::build(
		hitable** list,
		uint32_t num)
	{
		m_bvh.build(list, num);

		// Gather local-world matrix.
		{
			auto& shapes = const_cast<std::vector<transformable*>&>(transformable::getShapes());

			for (auto s : shapes) {
				auto& param = const_cast<aten::ShapeParameter&>(s->getParam());

				if (param.type == ShapeType::Instance) {
					aten::mat4 mtxL2W, mtxW2L;
					s->getMatrices(mtxL2W, mtxW2L);

					if (!mtxL2W.isIdentity()) {
						param.mtxid = (int)(s_mtxs.size() / 2);

						s_mtxs.push_back(mtxL2W);
						s_mtxs.push_back(mtxW2L);
					}
				}
			}
		}

		auto& bvhList = bvh::getBvhList();

		std::vector<std::vector<GPUBvhNodeEntry>> listBvhNode;
		listBvhNode.resize(bvhList.size() + 1);
		
		// Register to linear list to traverse bvhnode easily.
		auto root = m_bvh.getRoot();
		registerBvhNodeToLinearList(root, nullptr, nullptr, aten::mat4::Identity, listBvhNode[0]);

		for (int i = 0; i < bvhList.size(); i++) {
			auto bvh = bvhList[i];
			root = bvh->getRoot();

			hitable* parent = nullptr;

			// TODO
			// Find parent.
			for (auto it : s_nestedBvhMap) {
				auto& list = it.second;
				auto found = std::find(list.begin(), list.end(), bvh);

				if (found != list.end()) {
					// Found nested bvh.
					parent = it.first;
					break;
				}
			}

			registerBvhNodeToLinearList(root, nullptr, parent, aten::mat4::Identity, listBvhNode[i + 1]);
		}

		// Register bvh node for gpu.
		for (int i = 0; i < listBvhNode.size(); i++) {
			s_listGpuBvhNode.push_back(std::vector<GPUBvhNode>());

			// Leaves of nested bvh are primitive.
			// Index 0 is primiary tree, and Index N (N > 0) is nested tree.
			bool isPrimitiveLeaf = (i > 0);

			registerGpuBvhNode(isPrimitiveLeaf, listBvhNode[i], s_listGpuBvhNode[i]);
		}

		// Set traverse order for linear bvh.
		for (int i = 0; i < listBvhNode.size(); i++) {
			setOrderForLinearBVH(listBvhNode[i], s_listGpuBvhNode[i]);
		}

		//dump(s_listGpuBvhNode[1], "node.txt");
	}

	void GPUBvh::dump(std::vector<GPUBvhNode>& nodes, const char* path)
	{
		FILE* fp = nullptr;
		fopen_s(&fp, path, "wt");
		if (!fp) {
			AT_ASSERT(false);
			return;
		}

		for (const auto& n : nodes) {
			fprintf(fp, "%d %d %d %d (%.3f, %.3f, %.3f) (%.3f, %.3f, %.3f)\n", 
				(int)n.hit, (int)n.miss, (int)n.shapeid, (int)n.primid,
				n.boxmin.x, n.boxmin.y, n.boxmin.z,
				n.boxmax.x, n.boxmax.y, n.boxmax.z);
		}

		fclose(fp);
	}

	static inline bvhnode* getInternalNode(bvhnode* node, aten::mat4* mtxL2W = nullptr)
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

	void GPUBvh::registerBvhNodeToLinearList(
		bvhnode* root,
		bvhnode* parentNode,
		hitable* nestParent,
		const aten::mat4& mtxL2W,
		std::vector<GPUBvhNodeEntry>& listBvhNode)
	{
		if (!root) {
			return;
		}

		auto pnode = root;

		auto tmp = pnode;
		aten::mat4 mtxL2WForChild;
		pnode = getInternalNode(pnode, &mtxL2WForChild);

		if (pnode != tmp) {
			// Register nested bvh.
			hitable* parent = tmp->getItem();
			AT_ASSERT(parent->isInstance());

			// Register relation between instance and nested bvh.
			{
				auto child = const_cast<hitable*>(parent->getHasObject());

				// TODO
				auto obj = (object*)child;

				std::vector<accelerator*> accels;
				for (auto s : obj->shapes) {
					auto nestedBvh = s->getInternalAccelerator();
					accels.push_back(nestedBvh);
				}

				if (!accels.empty()) {
					s_nestedBvhMap.insert(std::pair<hitable*, std::vector<accelerator*>>(parent, accels));
				}
			}

			registerBvhNodeToLinearList(pnode, parentNode, parent, mtxL2WForChild, listBvhNode);
		}
		else {
			pnode->setParent(parentNode);
			pnode->setTraversalOrder((int)listBvhNode.size());
			listBvhNode.push_back(GPUBvhNodeEntry(pnode, nestParent, mtxL2W));

			if (pnode->isLeaf()) {
				auto item = pnode->getItem();
				auto externalId = item->getExtraId();

				if (externalId >= 0) {
					auto externalId = item->getExtraId() + 1;	// Index 0 is for main tree.
					pnode->setExternalId(externalId);
				}
			}

			bvhnode* pleft = pnode->getLeft();
			bvhnode* pright = pnode->getRight();

			registerBvhNodeToLinearList(pleft, pnode, nestParent, mtxL2W, listBvhNode);
			registerBvhNodeToLinearList(pright, pnode, nestParent, mtxL2W, listBvhNode);
		}
	}

	void GPUBvh::registerGpuBvhNode(
		bool isPrimitiveLeaf,
		std::vector<GPUBvhNodeEntry>& listBvhNode,
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

			gpunode.padding0 = -1.0f;

			if (node->isLeaf()) {
				hitable* item = node->getItem();
				auto internalObj = item->getHasObject();

				if (internalObj) {
					// This node is instance, so find index as internal object in instance.
					item = const_cast<hitable*>(internalObj);
				}

				gpunode.shapeid = (float)transformable::findShapeIdxAsHitable(item);

				if (gpunode.shapeid < 0) {
					if (nestParent) {
						gpunode.shapeid = (float)transformable::findShapeIdxAsHitable(nestParent);
					}
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

			gpunode.boxmax = bbox.maxPos();
			gpunode.boxmin = bbox.minPos();

			listGpuBvhNode.push_back(gpunode);
		}
	}

	void GPUBvh::setOrderForLinearBVH(
		std::vector<GPUBvhNodeEntry>& listBvhNode,
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
					bvhnode* left = getInternalNode(parent->getLeft());
					bvhnode* right = getInternalNode(parent->getRight());

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
								bvhnode* _left = getInternalNode(grandParent->getLeft());;
								bvhnode* _right = getInternalNode(grandParent->getRight());;

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

	bool GPUBvh::hit(
		const ray& r,
		real t_min, real t_max,
		Intersection& isect) const
	{
		return hit(0, s_listGpuBvhNode, r, t_min, t_max, isect);
	}

	bool GPUBvh::hit(
		int exid,
		std::vector<std::vector<GPUBvhNode>>& listGpuBvhNode,
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
			GPUBvhNode* node = nullptr;

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
						const auto& mtxW2L = s_mtxs[mtxid * 2 + 1];

						transformedRay = mtxW2L.applyRay(r);
					}
					else {
						transformedRay = r;
					}

					isHit = hit(
						node->exid,
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
