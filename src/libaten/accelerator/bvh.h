#pragma once

#include "scene/hitable.h"
#include "shape/tranformable.h"
#include "object/object.h"
#include "accelerator/accelerator.h"

#include <functional>

namespace aten {
	class transformable;

	class bvhnode {
		friend class bvh;

	public:
		bvhnode(bvhnode* parent) : m_parent(parent) {}
		virtual ~bvhnode() {}

	private:
		bvhnode(bvhnode* parent, hitable* item)
			: m_item(item), m_parent(parent)
		{}

	public:
		bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const;

		const aabb& getBoundingbox() const
		{
			if (m_item) {
				return m_item->getBoundingbox();
			}
			return m_aabb;
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

	protected:
		bvhnode* m_left{ nullptr };
		bvhnode* m_right{ nullptr };

		bvhnode* m_parent{ nullptr };

		aabb m_aabb;

		hitable* m_item{ nullptr };

		int m_traverseOrder{ -1 };
		int m_externalId{ -1 };
	};

	//////////////////////////////////////////////

	class bvh : public accelerator {
		friend class bvhnode;
		friend class accelerator;
		friend class GPUBvh;

	public:
		bvh() {}
		virtual ~bvh() {}

	public:
		virtual void build(
			hitable** list,
			uint32_t num) override;

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
				aten::hitable* parent = original->getItem();
				AT_ASSERT(parent->isInstance());

				// Register relation between instance and nested bvh.
				auto child = const_cast<hitable*>(parent->getHasObject());

				// TODO
				auto obj = (AT_NAME::object*)child;

				std::vector<aten::accelerator*> accels;
				auto nestedBvh = obj->getInternalAccelerator();
				accels.push_back(nestedBvh);

				auto found = std::find(listBvh.begin(), listBvh.end(), nestedBvh);
				if (found == listBvh.end()) {
					listBvh.push_back(nestedBvh);
				}

				if (!accels.empty()) {
					nestedBvhMap.insert(std::pair<aten::hitable*, std::vector<aten::accelerator*>>(parent, accels));
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

	private:
		static bool hit(
			const bvhnode* root,
			const ray& r,
			real t_min, real t_max,
			Intersection& isect);

		static void buildBySAH(
			bvhnode* root,
			hitable** list,
			uint32_t num);

	protected:
		bvhnode* m_root{ nullptr };
	};
}
