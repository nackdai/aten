#include "accelerator/bvh.h"
#include "accelerator/bvh_node.h"
#include "geometry/transformable.h"
#include "geometry/PolygonObject.h"

#include <random>
#include <vector>

// NOTE
// Fast, Effective BVH Updates for Animated Scenes.
// https://hwrt.cs.utah.edu/papers/hwrt_rotations.pdf

namespace aten
{
    void bvhnode::itemChanged(const hitable* sender)
    {
        AT_ASSERT(item_ == sender);

        if (bvh_) {
            auto oldBox = aabb_;

            aabb_ = sender->GetBoundingbox();

            // TODO
            bool isEqual = (memcmp(&oldBox, &aabb_, sizeof(aabb_)) == 0);

            if (!isEqual) {
                if (parent_) {
                    refitChildren(parent_, true);
                }

                bvh_->AddToRefit(this);
            }
        }
    }

    void bvhnode::refitChildren(bvhnode* node, bool propagate)
    {
        do {
            auto oldbox = node->GetBoundingbox();

            const auto& left = node->getLeft();
            const auto& right = node->getRight();

            // Start with the left box.
            auto newbox = left->GetBoundingbox();

            // Expand.
            if (right) {
                newbox.expand(right->GetBoundingbox());
            }

            // Set new box.
            node->setBoundingBox(newbox);

            // Walk up the tree.
            node = node->getParent();
        } while (propagate && node != nullptr);
    }

    inline float computeSurfaceArea(const bvhnode* node)
    {
        if (!node) {
            return float(0);
        }

        auto ret = node->GetBoundingbox().computeSurfaceArea();
        return ret;
    }

    inline float computeSurfaceArea(const std::shared_ptr<bvhnode>& node)
    {
        return computeSurfaceArea(node.get());
    }

    inline bool checkInvalidRotateChildToGrandChild(
        const std::shared_ptr<const bvhnode>& node0,
        const std::shared_ptr<const bvhnode>& node1)
    {
        bool b0 = (node0 == nullptr);
        bool b1 = (node1 == nullptr);
        bool b2 = (node1 && node1->isLeaf());
        bool b3 = (node1 && (node1->getLeft() == nullptr || node1->getRight() == nullptr));

        return b0 || b1 || b2 || b3;
    }

    inline bool checkInvalidRotateGrandChildToGrandChild(
        const std::shared_ptr<const bvhnode>& node0,
        const std::shared_ptr<const bvhnode>& node1)
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
        AT_ASSERT(bvh_ == bvh);

        if (bvh->GetRoot() == this) {
            return;
        }

        auto left = left_;
        auto right = right_;

        // If we are not a grandparent, then we can't rotate, so queue our parent and bail out.
        if ((left && left->isLeaf())
            && (right && right->isLeaf()))
        {
            if (parent_) {
                bvh->AddToRefit(parent_);
                return;
            }
        }

        // The list of all candidate rotations, from "Fast, Effective BVH Updates for Animated Scenes", Figure 1.
        enum Rot : int32_t {
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
            float sah;
            Rot rot;

            Opt() {}
            Opt(float _sah, Rot _rot) : sah(_sah), rot(_rot) {}
        };

        // For each rotation, check that there are grandchildren as necessary (aka not a leaf)
        // then compute total SAH cost of our branches after the rotation.
        auto sa = computeSurfaceArea(left) + computeSurfaceArea(right);

        std::vector<Opt> opts(Rot::Num);

        for (int32_t r = Rot::None; r < Rot::Num; r++) {
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
                    auto box = left->GetBoundingbox();
                    box.expand(right->getRight()->GetBoundingbox());

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
                    auto box = left->GetBoundingbox();
                    box.expand(right->getLeft()->GetBoundingbox());

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
                    auto box = right->GetBoundingbox();
                    box.expand(left->getRight()->GetBoundingbox());

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
                    auto box = right->GetBoundingbox();
                    box.expand(left->getLeft()->GetBoundingbox());

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
                    auto box0 = right->getRight()->GetBoundingbox();
                    box0.expand(left->getRight()->GetBoundingbox());

                    auto box1 = right->getLeft()->GetBoundingbox();
                    box1.expand(left->getLeft()->GetBoundingbox());

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
                    auto box0 = right->getLeft()->GetBoundingbox();
                    box0.expand(left->getRight()->GetBoundingbox());

                    auto box1 = left->getLeft()->GetBoundingbox();
                    box1.expand(right->getRight()->GetBoundingbox());

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
            if (parent_) {
                bvh->AddToRefit(parent_);
            }
        }
        else {
            if (parent_) {
                bvh->AddToRefit(parent_);
            }

            auto s = (sa - bestRot.sah) / sa;

            if (s < float(0.3)) {
                // The benefit is not worth the cost
                return;
            }
            else {
                // In order to swap we need to:
                //    1. swap the node locations
                //    2. update the depth (if child-to-grandchild)
                //    3. update the parent pointers
                //    4. refit the boundary box
                decltype(left) swap;

                switch (bestRot.rot) {
                case Rot::None:
                    break;

                // child to grandchild rotations
                case Rot::L_RL:
                    swap = left;
                    left = right->left_;
                    left->parent_ = this;
                    right->left_ = swap;
                    swap->parent_ = right.get();
                    refitChildren(right, false);
                    break;
                case Rot::L_RR:
                    swap = left;
                    left = right->right_;
                    left->parent_ = this;
                    right->right_ = swap;
                    swap->parent_ = right.get();
                    refitChildren(right, false);
                    break;
                case Rot::R_LL:
                    swap = right;
                    right = left->left_;
                    right->parent_ = this;
                    left->left_ = swap;
                    swap->parent_ = left.get();
                    refitChildren(left, false);
                    break;
                case Rot::R_LR:
                    swap = right;
                    right = left->right_;
                    right->parent_ = this;
                    left->right_ = swap;
                    swap->parent_ = left.get();
                    refitChildren(left, false);
                    break;

                // grandchild to grandchild rotations
                case Rot::LL_RR:
                    swap = left->left_;
                    left->left_ = right->right_;
                    right->right_ = swap;
                    left->left_->parent_ = left.get();
                    swap->parent_ = right.get();
                    refitChildren(left, false);
                    refitChildren(right, false);
                    break;
                case Rot::LL_RL:
                    swap = left->left_;
                    left->left_ = right->left_;
                    right->left_ = swap;
                    left->left_->parent_ = left.get();
                    swap->parent_ = right.get();
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
                this->setDepth(depth_);
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
        sweepNodes.reserve(refit_nodes_.size());

        while (!refit_nodes_.empty()) {
            int32_t maxdepth = -1;

            for (const auto node : refit_nodes_) {
                maxdepth = std::max<int32_t>(node->getDepth(), maxdepth);
            }

            sweepNodes.clear();

            auto it = refit_nodes_.begin();

            while (it != refit_nodes_.end()) {
                auto node = *it;
                int32_t depth = node->getDepth();

                if (maxdepth == depth) {
                    sweepNodes.push_back(node);
                    it = refit_nodes_.erase(it);
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
