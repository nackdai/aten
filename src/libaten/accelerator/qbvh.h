#pragma once

#include "accelerator/bvh.h"

namespace aten {
	struct QbvhNode {
		union {
			aten::vec4 p0;
			struct {
				float leftChildrenIdx;
				float isLeaf;
				float numChildren;

				// TODO
				// BVHでは4つのリーフを１つのリーフで扱うので、shape(sphere etc)とprimitive(triangle)が入り乱れるので、その判定フラグにする.
				// 現時点ではprimitive(triangle)のみ.
				float padding;
			};
		};

		union {
			aten::vec4 p1;
			struct {
				float shapeid;	///< Object index.
				float primid;	///< Triangle index.
				float exid;		///< External bvh index.
				float meshid;	///< Mesh id.
			};
		};

		union {
			aten::vec4 p2;
			struct {
				float shapeidx[4];
			};
		};
		union {
			aten::vec4 p3;
			struct {
				float primidx[4];
			};
		};
		union {
			aten::vec4 p4;
			struct {
				float meshidx[4];
			};
		};

		aten::vec4 bminx;
		aten::vec4 bmaxx;
		aten::vec4 bminy;
		aten::vec4 bmaxy;
		aten::vec4 bminz;
		aten::vec4 bmaxz;

		QbvhNode()
		{
			isLeaf = false;
			numChildren = 0;

			shapeid = -1;
			primid = -1;
			meshid = -1;
			exid = -1;
		}
		QbvhNode(const QbvhNode& rhs)
		{
			p0 = rhs.p0;
			p1 = rhs.p1;
			p2 = rhs.p2;
			p3 = rhs.p3;
			p4 = rhs.p4;

			bminx = rhs.bminx;
			bmaxx = rhs.bmaxx;

			bminy = rhs.bminy;
			bmaxy = rhs.bmaxy;

			bminz = rhs.bminz;
			bmaxz = rhs.bmaxz;
		}
	};

	class transformable;

	class qbvh : public accelerator {
	public:
		qbvh() {}
		virtual ~qbvh() {}

	public:
		virtual void build(
			hitable** list,
			uint32_t num) override final;

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const override;

		std::vector<std::vector<QbvhNode>>& getNodes()
		{
			return m_listQbvhNode;
		}
		std::vector<aten::mat4>& getMatrices()
		{
			return m_mtxs;
		}

	private:
		struct BvhNode {
			bvhnode* node;
			hitable* nestParent;
			aten::mat4 mtxL2W;

			BvhNode(bvhnode* n, hitable* p, const aten::mat4& m)
				: node(n), nestParent(p), mtxL2W(m)
			{}
		};

		void registerBvhNodeToLinearList(
			bvhnode* root,
			bvhnode* parentNode,
			hitable* nestParent,
			const aten::mat4& mtxL2W,
			std::vector<BvhNode>& listBvhNode,
			std::vector<accelerator*>& listBvh,
			std::map<hitable*, std::vector<accelerator*>>& nestedBvhMap);

		uint32_t convertFromBvh(
			bool isPrimitiveLeaf,
			std::vector<BvhNode>& listBvhNode,
			std::vector<QbvhNode>& listQbvhNode);

		void setQbvhNodeLeafParams(
			bool isPrimitiveLeaf,
			const BvhNode& bvhNode,
			QbvhNode& qbvhNode);

		void fillQbvhNode(
			QbvhNode& qbvhNode,
			std::vector<BvhNode>& listBvhNode,
			int children[4],
			int numChildren);

		int getChildren(
			std::vector<BvhNode>& listBvhNode,
			int bvhNodeIdx,
			int children[4]);

		bool hit(
			int exid,
			const std::vector<std::vector<QbvhNode>>& listQbvhNode,
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const;

	private:
		bvh m_bvh;

		int m_exid{ 1 };

		std::vector<std::vector<QbvhNode>> m_listQbvhNode;
		std::vector<aten::mat4> m_mtxs;
	};
}
