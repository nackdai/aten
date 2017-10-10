#include "accelerator/bvh.h"
#include "shape/tranformable.h"
#include "object/object.h"

#include <random>
#include <vector>

#define TEST_NODE_LIST
//#pragma optimize( "", off)

namespace aten {
	void bvh::collectNodes(
		std::vector<std::vector<BVHNode>>& nodes,
		std::vector<aten::mat4>& mtxs) const
	{
		std::vector<std::vector<bvhnode*>> nodeList(1);
		registerToList(m_root, 0, nodeList);

		nodes.resize(nodeList.size());

		collectNodes(
			m_root,
			0,
			nodes,
			nullptr,
			mat4::Identity);

		auto& shapes = const_cast<std::vector<transformable*>&>(transformable::getShapes());

		for (auto s : shapes) {
			auto& param = const_cast<aten::ShapeParameter&>(s->getParam());

			if (param.type == ShapeType::Instance) {
				aten::mat4 mtxL2W, mtxW2L;
				s->getMatrices(mtxL2W, mtxW2L);

				if (!mtxL2W.isIdentity()) {
					param.mtxid = mtxs.size() / 2;

					mtxs.push_back(mtxL2W);
					mtxs.push_back(mtxW2L);
				}
			}
		}

		// Specify hit/miss link.

		// NOTE
		// http://www.ci.i.u-tokyo.ac.jp/~hachisuka/tdf2015.pdf
		// Threaded BVH.

		for (int i = 0; i < nodeList.size(); i++) {
			auto num = nodeList[i].size();

			for (int n = 0; n < num; n++) {
				auto pnode = nodeList[i][n];
				auto& node = nodes[i][n];

				bvhnode* next = nullptr;
				if (n + 1 < num) {
					next = nodeList[i][n + 1];
				}

				if (pnode->isLeaf()) {
					// Hit/Miss.
					// Always the next node in the array.
					if (next) {
						node.hit = (float)next->m_traverseOrder;
						node.miss = (float)next->m_traverseOrder;
					}
					else {
						node.hit = -1;
						node.miss = -1;
					}
				}
				else {
					// Hit.
					// Always the next node in the array.
					if (next) {
						node.hit = next->m_traverseOrder;
					}
					else {
						node.hit = -1;
					}

					// Miss.

					// Search the parent.
					bvhnode* parent = (node.parent >= 0
						? nodeList[i][node.parent]
						: nullptr);
					
					if (parent) {
						bvhnode* left = nullptr;
						bvhnode* right = nullptr;

						parent->getNodes(left, right);

						bool isLeft = (left == pnode);

						if (isLeft) {
							// Internal, left: sibling node.
							auto sibling = right;
							isLeft = (sibling != nullptr);

							if (isLeft) {
								node.miss = (float)sibling->m_traverseOrder;
							}
						}

						bvhnode* curParent = parent;

						if (!isLeft) {
							// Internal, right: parentfs sibling node (until it exists) .
							for (;;) {
								// Search the grand parent.
								const auto& parentNode = nodes[i][curParent->m_traverseOrder];
								bvhnode* grandParent = (parentNode.parent >= 0
									? nodeList[i][parentNode.parent]
									: nullptr);

								if (grandParent) {
									bvhnode* _left = nullptr;
									bvhnode* _right = nullptr;

									grandParent->getNodes(_left, _right);

									auto sibling = _right;
									if (sibling) {
										if (sibling != curParent) {
											node.miss = (float)sibling->m_traverseOrder;
											break;
										}
									}
								}
								else {
									node.miss = -1;
									break;
								}

								curParent = grandParent;
							}
						}
					}
					else {
						node.miss = -1;
					}
				}
			}
		}
	}

	void bvh::registerToList(
		bvhnode* root,
		const int idx,
		std::vector<std::vector<bvhnode*>>& nodeList)
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

			pnode->registerToList(idx, nodeList);

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

	void bvh::collectNodes(
		bvhnode* root,
		const int idx,
		std::vector<std::vector<BVHNode>>& nodes,
		const transformable* instanceParent,
		const aten::mat4& mtxL2W)
	{
		collectNodes(root, nullptr, idx, nodes, instanceParent, mtxL2W);
	}

	void bvh::collectNodes(
		bvhnode* pnode,
		const bvhnode* parent,
		const int idx,
		std::vector<std::vector<BVHNode>>& nodes,
		const transformable* instanceParent,
		const aten::mat4& mtxL2W)
	{
		if (!pnode) {
			return;
		}

		BVHNode node;

		if (parent) {
			node.parent = (float)parent->m_traverseOrder;
		}

		if (pnode->setBVHNodeParam(node, parent, idx, nodes, instanceParent, mtxL2W)) {
			// NOTE
			// Differed spcification hit/miss link.

			nodes[idx].push_back(node);
		}

		collectNodes(pnode->m_left, pnode, idx, nodes, instanceParent, mtxL2W);
		collectNodes(pnode->m_right, pnode, idx, nodes, instanceParent, mtxL2W);
	}

	void bvh::dumpCollectedNodes(std::vector<BVHNode>& nodes, const char* path)
	{
		FILE* fp = nullptr;
		fopen_s(&fp, path, "wt");
		if (!fp) {
			AT_ASSERT(false);
			return;
		}

		for (const auto& n : nodes) {
			fprintf(fp, "%d %d %d %d\n", (int)n.hit, (int)n.miss, (int)n.shapeid, (int)n.primid);
		}

		fclose(fp);
	} 
}
