#pragma once

#include "scene/hitable.h"
#include "geometry/tranformable.h"
#include "geometry/object.h"
#include "accelerator/accelerator.h"

#include <functional>
#include <stack>

namespace aten {
	class transformable;

	class bvhnode {
		friend class bvh;

	public:
		bvhnode(bvhnode* parent)
			: m_parent(parent)
		{
			m_children[0] = m_children[1] = m_children[2] = m_children[3] = nullptr;
		}
		virtual ~bvhnode() {}

	private:
		bvhnode(bvhnode* parent, hitable* item)
			: m_parent(parent)
		{
			m_children[0] = m_children[1] = m_children[2] = m_children[3] = nullptr;
			m_item = item;
		}

	public:
		bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const;

		const aabb& getBoundingbox() const
		{
#if 0
			if (m_item) {
				return m_item->getBoundingbox();
			}
			return m_aabb;
#else
			return m_aabb;
#endif
		}
		void setBoundingBox(const aabb& bbox)
		{
			m_aabb = bbox;
		}

		bool isLeaf() const
		{
			return (!m_left && !m_right);
		}

		bvhnode* getLeft()
		{
			return m_left;
		}
		bvhnode* getRight()
		{
			return m_right;
		}

		void setParent(bvhnode* parent)
		{
			m_parent = parent;
		}
		bvhnode* getParent()
		{
			return m_parent;
		}

		hitable* getItem()
		{
			return m_item;
		}

		int getTraversalOrder() const
		{
			return m_traverseOrder;
		}
		void setTraversalOrder(int order)
		{
			m_traverseOrder = order;
		}

		int getExternalId() const
		{
			return m_externalId;
		}
		void setExternalId(int exid)
		{
			m_externalId = exid;
		}

		int getChildrenNum() const
		{
			return m_childrenNum;
		}
		void setChildrenNum(int num)
		{
			AT_ASSERT((0 <= num) && (num <= 4));
			m_childrenNum = num;
		}

		hitable** getChildren()
		{
			return m_children;
		}
		void registerChild(hitable* child, int idx)
		{
			m_children[idx] = child;
		}

		void setDepth(int depth)
		{
			m_depth = depth;
		}
		void setDepthToChildren(int depth)
		{
			m_depth = depth;
			if (m_left) {
				m_left->setDepthToChildren(depth + 1);
			}
			if (m_right) {
				m_right->setDepthToChildren(depth + 1);
			}
		}
		int getDepth() const
		{
			return m_depth;
		}

	private:
		void setIsCandidate(bool c)
		{
			m_isCandidate = c;
		}
		bool isCandidate() const
		{
			return m_isCandidate;
		}

		void drawAABB(
			aten::hitable::FuncDrawAABB func,
			const aten::mat4& mtxL2W) const;

	protected:
		bvhnode* m_left{ nullptr };
		bvhnode* m_right{ nullptr };

		bvhnode* m_parent{ nullptr };

		aabb m_aabb;

		union {
			struct {
				hitable* m_item;
				hitable* padding[3];
			};
			hitable* m_children[4];
		};

		int m_traverseOrder{ -1 };
		int m_externalId{ -1 };
		int m_childrenNum{ 0 };

		int m_depth{ 0 };

		bool m_isCandidate{ false };
	};

	//////////////////////////////////////////////

	class bvh : public accelerator {
		friend class bvhnode;
		friend class accelerator;
		friend class ThreadedBVH;

	public:
		bvh() : accelerator(AccelType::Bvh) {}
		virtual ~bvh() {}

	public:
		virtual void build(
			hitable** list,
			uint32_t num,
			aabb* bbox = nullptr) override;

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const override;

		virtual const aabb& getBoundingbox() const override
		{
			if (m_root) {
				return std::move(m_root->getBoundingbox());
			}
			return std::move(aabb());
		}

		bvhnode* getRoot()
		{
			return m_root;
		}

		virtual accelerator::ResultIntersectTestByFrustum intersectTestByFrustum(const frustum& f) override final;

		static bvhnode* getInternalNode(bvhnode* node, aten::mat4* mtxL2W = nullptr);

		template <typename _T>
		static void registerBvhNodeToLinearList(
			aten::bvhnode* root,
			aten::bvhnode* parentNode,
			aten::hitable* nestParent,
			const aten::mat4& mtxL2W,
			std::vector<_T>& listBvhNode,
			std::vector<aten::accelerator*>& listBvh,
			std::map<hitable*, std::vector<aten::accelerator*>>& nestedBvhMap,
			std::function<void(std::vector<_T>&, aten::bvhnode*, aten::hitable*, const aten::mat4&)> funcRegisterToList,
			std::function<void(aten::bvhnode*)> funcIfInstanceNode)
		{
			if (!root) {
				return;
			}

			auto pnode = root;

			auto original = pnode;
			aten::mat4 mtxL2WForChild;
			pnode = getInternalNode(original, &mtxL2WForChild);

			if (pnode != original) {
				{
					original->setParent(parentNode);
					original->setTraversalOrder((int)listBvhNode.size());
					funcRegisterToList(listBvhNode, original, nestParent, mtxL2W);

					if (funcIfInstanceNode) {
						funcIfInstanceNode(original);
					}
				}

				// Register nested bvh.
				aten::hitable* originalItem = original->getItem();
				AT_ASSERT(originalItem->isInstance());

				// Register relation between instance and nested bvh.
				auto internalItem = const_cast<hitable*>(originalItem->getHasObject());

				// TODO
				auto obj = (AT_NAME::object*)internalItem;

				std::vector<aten::accelerator*> accels;
				auto nestedBvh = obj->getInternalAccelerator();
				accels.push_back(nestedBvh);

				auto found = std::find(listBvh.begin(), listBvh.end(), nestedBvh);
				if (found == listBvh.end()) {
					listBvh.push_back(nestedBvh);
				}

				if (!accels.empty()) {
					nestedBvhMap.insert(std::pair<aten::hitable*, std::vector<aten::accelerator*>>(originalItem, accels));
				}
			}
			else {
				pnode->setParent(parentNode);
				pnode->setTraversalOrder((int)listBvhNode.size());
				funcRegisterToList(listBvhNode, pnode, nestParent, mtxL2W);

				aten::bvhnode* pleft = pnode->getLeft();
				aten::bvhnode* pright = pnode->getRight();

				registerBvhNodeToLinearList(pleft, pnode, nestParent, mtxL2W, listBvhNode, listBvh, nestedBvhMap, funcRegisterToList, funcIfInstanceNode);
				registerBvhNodeToLinearList(pright, pnode, nestParent, mtxL2W, listBvhNode, listBvh, nestedBvhMap, funcRegisterToList, funcIfInstanceNode);
			}
		}

		virtual void drawAABB(
			aten::hitable::FuncDrawAABB func,
			const aten::mat4& mtxL2W) override;

	private:
		static bool hit(
			const bvhnode* root,
			const ray& r,
			real t_min, real t_max,
			Intersection& isect);

		static void buildBySAH(
			bvhnode* root,
			hitable** list,
			uint32_t num,
			int depth = 0,
			bvhnode* parent = nullptr);

		struct Candidate {
			bvhnode* node{ nullptr };
			bvhnode* instanceNode{ nullptr };

			Candidate(bvhnode* n, bvhnode* i = nullptr)
			{
				node = n;
				instanceNode = i;
			}
		};

		bool findCandidates(
			bvhnode* node,
			bvhnode* instanceNode,
			const frustum& f,
			std::stack<Candidate>* stack);

		bvhnode* traverse(
			bvhnode* root,
			const frustum& f);

	protected:
		bvhnode* m_root{ nullptr };
	};
}
