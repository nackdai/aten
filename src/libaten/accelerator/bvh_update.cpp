#include "accelerator/bvh.h"
#include "geometry/transformable.h"
#include "geometry/object.h"

#include <random>
#include <vector>

// NOTE
// Fast, Effective BVH Updates for Animated Scenes.
// http://www.cs.utah.edu/~thiago/papers/rotations.pdf

namespace aten
{
    void bvhnode::itemChanged(hitable* sender)
    {
        AT_ASSERT(m_item == sender);

        if (m_bvh) {
            auto oldBox = m_aabb;

            m_aabb = sender->getBoundingbox();

            // TODO
            bool isEqual = (memcmp(&oldBox, &m_aabb, sizeof(m_aabb)) == 0);

            if (!isEqual) {
                if (m_parent) {
                    refitChildren(m_parent, true);
                }

                m_bvh->addToRefit(this);
            }
        }
    }

    void bvhnode::refitChildren(bvhnode* node, bool propagate)
    {
        do {
            auto oldbox = node->getBoundingbox();

            auto left = node->getLeft();
            auto right = node->getRight();

            // Start with the left box.
            auto newbox = left->getBoundingbox();

            // Expand.
            if (right) {
                newbox.expand(right->getBoundingbox());
            }

            // Set new box.
            node->setBoundingBox(newbox);

            // Walk up the tree.
            node = node->getParent();
        } while (propagate && node != nullptr);
    }

    inline real computeSurfaceArea(bvhnode* node)
    {
        if (!node) {
            return real(0);
        }

        auto ret = node->getBoundingbox().computeSurfaceArea();
        return ret;
    }

    inline bool checkInvalidRotateChildToGrandChild(
        bvhnode* node0,
        bvhnode* node1)
    {
        bool b0 = (node0 == nullptr);
        bool b1 = (node1 == nullptr);
        bool b2 = (node1 && node1->isLeaf());
        bool b3 = (node1 && (node1->getLeft() == nullptr || node1->getRight() == nullptr));

        return b0 || b1 || b2 || b3;
    }

    inline bool checkInvalidRotateGrandChildToGrandChild(
        bvhnode* node0,
        bvhnode* node1)
    {
        bool b0 = (node0 == nullptr);
        bool b1 = (node1 == nullptr);
        bool b2 = (node0 && node0->isLeaf()) || (node1 && node1->isLeaf());
        bool b3 = (node0 && (node0->getLeft() == nullptr || node0->getRight() == nullptr));
        bool b4 = (node1 && (node1->getLeft() == nullptr || node1->getRight() == nullptr));

        return b0 || b1 || b2 || b3 || b4;
    }

    void bvhnode::tryRotate(bvh* bvh)
    {
        AT_ASSERT(m_bvh == bvh);

        if (bvh->getRoot() == this) {
            return;
        }

        auto left = m_left;
        auto right = m_right;

        // If we are not a grandparent, then we can't rotate, so queue our parent and bail out.
        if ((left && left->isLeaf())
            && (right && right->isLeaf()))
        {
            if (m_parent) {
                bvh->addToRefit(m_parent);
                return;
            }
        }

        // The list of all candidate rotations, from "Fast, Effective BVH Updates for Animated Scenes", Figure 1.
        enum Rot : int {
            None,

            // child to grandchild rotations.

            L_RL,    // Left  <-> RightLeft,
            L_RR,    // Left  <-> RightRight,
            R_LL,    // Right <-> LeftLeft,
            R_LR,    // Right <-> LeftRight,

            // grandchild to grandchild rotations.

            LL_RR,    // LeftLeft <-> RightRight,
            LL_RL,    // LeftLeft <-> RightLeft,

            Num,
        };

        struct Opt {
            real sah;
            Rot rot;

            Opt() {}
            Opt(real _sah, Rot _rot) : sah(_sah), rot(_rot) {}
        };

        // For each rotation, check that there are grandchildren as necessary (aka not a leaf)
        // then compute total SAH cost of our branches after the rotation.
        auto sa = computeSurfaceArea(left) + computeSurfaceArea(right);

        std::vector<Opt> opts(Rot::Num);

        for (int r = Rot::None; r < Rot::Num; r++) {
            switch (r) {
            case Rot::None:
            {
                opts[r] = Opt(sa, Rot::None);
                break;
            }

            // child to grandchild rotations
            case Rot::L_RL:
            {
                // Left <-> RightLeft,

                if (checkInvalidRotateChildToGrandChild(left, right))
                {
                    opts[r] = Opt(AT_MATH_INF, Rot::None);
                }
                else {
                    auto box = left->getBoundingbox();
                    box.expand(right->getRight()->getBoundingbox());

                    auto s = computeSurfaceArea(right->getLeft()) + box.computeSurfaceArea();

                    opts[r] = Opt(s, (Rot)r);
                }
                break;
            }
            case Rot::L_RR:
            {
                // Left <-> RightRight,

                if (checkInvalidRotateChildToGrandChild(left, right))
                {
                    opts[r] = Opt(AT_MATH_INF, Rot::None);
                }
                else {
                    auto box = left->getBoundingbox();
                    box.expand(right->getLeft()->getBoundingbox());

                    auto s = computeSurfaceArea(right->getRight()) + box.computeSurfaceArea();

                    opts[r] = Opt(s, (Rot)r);
                }

                break;
            }
            case Rot::R_LL:
            {
                // Right <-> LeftLeft,

                if (checkInvalidRotateChildToGrandChild(right, left)) {
                    opts[r] = Opt(AT_MATH_INF, Rot::None);
                }
                else {
                    auto box = right->getBoundingbox();
                    box.expand(left->getRight()->getBoundingbox());

                    auto s = computeSurfaceArea(left->getLeft()) + box.computeSurfaceArea();

                    opts[r] = Opt(s, (Rot)r);
                }

                break;
            }
            case Rot::R_LR:
            {
                // Right <-> LeftRight,

                if (checkInvalidRotateChildToGrandChild(right, left)) {
                    opts[r] = Opt(AT_MATH_INF, Rot::None);
                }
                else {
                    auto box = right->getBoundingbox();
                    box.expand(left->getLeft()->getBoundingbox());

                    auto s = computeSurfaceArea(left->getRight()) + box.computeSurfaceArea();

                    opts[r] = Opt(s, (Rot)r);
                }

                break;
            }

            // grandchild to grandchild rotations
            case Rot::LL_RR:
            {
                // LeftLeft <-> RightRight,

                if (checkInvalidRotateGrandChildToGrandChild(left, right)) {
                    opts[r] = Opt(AT_MATH_INF, Rot::None);
                }
                else {
                    auto box0 = right->getRight()->getBoundingbox();
                    box0.expand(left->getRight()->getBoundingbox());

                    auto box1 = right->getLeft()->getBoundingbox();
                    box1.expand(left->getLeft()->getBoundingbox());

                    auto s = box0.computeSurfaceArea() + box1.computeSurfaceArea();

                    opts[r] = Opt(s, (Rot)r);
                }

                break;
            }
            case Rot::LL_RL:
            {
                // LeftLeft <-> RightLeft,
                if (checkInvalidRotateGrandChildToGrandChild(left, right)) {
                    opts[r] = Opt(AT_MATH_INF, Rot::None);
                }
                else {
                    auto box0 = right->getLeft()->getBoundingbox();
                    box0.expand(left->getRight()->getBoundingbox());

                    auto box1 = left->getLeft()->getBoundingbox();
                    box1.expand(right->getRight()->getBoundingbox());

                    auto s = box0.computeSurfaceArea() + box1.computeSurfaceArea();

                    opts[r] = Opt(s, (Rot)r);
                }

                break;
            }
            default:
                AT_ASSERT(false);
                break;
            }
        }

        Opt bestRot(AT_MATH_INF, Rot::None);
        for (const auto& o : opts) {
            bestRot = (o.sah < bestRot.sah ? o : bestRot);
        }

        // Perform the best rotation.
        if (bestRot.rot == Rot::None) {
            // If the best rotation is no-rotation... we check our parents anyhow..
            if (m_parent) {
                bvh->addToRefit(m_parent);
            }
        }
        else {
            if (m_parent) {
                bvh->addToRefit(m_parent);
            }

            auto s = (sa - bestRot.sah) / sa;

            if (s < real(0.3)) {
                // The benefit is not worth the cost
                return;
            }
            else {
                // In order to swap we need to:
                //    1. swap the node locations
                //    2. update the depth (if child-to-grandchild)
                //    3. update the parent pointers
                //    4. refit the boundary box
                bvhnode* swap = nullptr;

                switch (bestRot.rot) {
                case Rot::None:
                    break;

                // child to grandchild rotations
                case Rot::L_RL:
                    swap = left;
                    left = right->m_left;
                    left->m_parent = this;
                    right->m_left = swap;
                    swap->m_parent = right;
                    refitChildren(right, false);
                    break;
                case Rot::L_RR:
                    swap = left;
                    left = right->m_right;
                    left->m_parent = this;
                    right->m_right = swap;
                    swap->m_parent = right;
                    refitChildren(right, false);
                    break;
                case Rot::R_LL:
                    swap = right;
                    right = left->m_left;
                    right->m_parent = this;
                    left->m_left = swap;
                    swap->m_parent = left;
                    refitChildren(left, false);
                    break;
                case Rot::R_LR:
                    swap = right;
                    right = left->m_right;
                    right->m_parent = this;
                    left->m_right = swap;
                    swap->m_parent = left;
                    refitChildren(left, false);
                    break;

                // grandchild to grandchild rotations
                case Rot::LL_RR:
                    swap = left->m_left;
                    left->m_left = right->m_right;
                    right->m_right = swap;
                    left->m_left->m_parent = left;
                    swap->m_parent = right;
                    refitChildren(left, false);
                    refitChildren(right, false);
                    break;
                case Rot::LL_RL:
                    swap = left->m_left;
                    left->m_left = right->m_left;
                    right->m_left = swap;
                    left->m_left->m_parent = left;
                    swap->m_parent = right;
                    refitChildren(left, false);
                    refitChildren(right, false);
                    break;

                default:
                    AT_ASSERT(false);
                    break;
                }
            }

            // Fix the depths if necessary....
            switch (bestRot.rot) {
            case Rot::L_RL:
            case Rot::L_RR:
            case Rot::R_LL:
            case Rot::R_LR:
                this->setDepth(m_depth);
                this->propageteDepthToChildren();
                break;
            case Rot::None:
            case Rot::LL_RR:
            case Rot::LL_RL:
            case Rot::Num:
                break;
            }
        }
    }

    void bvh::update(const context& ctxt)
    {
        std::vector<bvhnode*> sweepNodes;
        sweepNodes.reserve(m_refitNodes.size());

        while (!m_refitNodes.empty()) {
            int maxdepth = -1;

            for (const auto node : m_refitNodes) {
                maxdepth = std::max<int>(node->getDepth(), maxdepth);
            }

            sweepNodes.clear();

            auto it = m_refitNodes.begin();

            while (it != m_refitNodes.end()) {
                auto node = *it;
                int depth = node->getDepth();

                if (maxdepth == depth) {
                    sweepNodes.push_back(node);
                    it = m_refitNodes.erase(it);
                }
                else {
                    it++;
                }
            }

            for (auto node : sweepNodes) {
                node->tryRotate(this);
            }
        }
    }
}
