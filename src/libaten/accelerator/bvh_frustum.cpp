#include <random>
#include <vector>

#include "accelerator/bvh.h"
#include "geometry/transformable.h"
#include "geometry/object.h"

namespace aten
{
    accelerator::ResultIntersectTestByFrustum bvh::intersectTestByFrustum(const frustum& f)
    {
        std::stack<Candidate> stack[2];

        accelerator::ResultIntersectTestByFrustum result;

        findCandidates(m_root, nullptr, f, stack);

        bvhnode* candidate = nullptr;

        while (!stack[1].empty()) {
            auto c = stack[1].top();
            stack[1].pop();

            auto instanceNode = c.instanceNode;

            bvhnode* node = c.node;
            int exid = 0;

            aten::mat4 mtxW2L;

            if (instanceNode) {
                auto n = getNestedNode(instanceNode, &mtxW2L);

#if 0
                aten::hitable* originalItem = n->getItem();
                AT_ASSERT(originalItem->isInstance());

                // Register relation between instance and nested bvh.
                auto internalItem = const_cast<hitable*>(originalItem->getHasObject());

                // TODO
                auto obj = (AT_NAME::object*)internalItem;

                auto nestedBvh = (bvh*)obj->getInternalAccelerator();
#endif

                mtxW2L.invert();

                exid = instanceNode->getExternalId();
            }

            candidate = node;

            auto transformedFrustum = f;
            transformedFrustum.transform(mtxW2L);

            auto n = traverse(node, transformedFrustum);

            if (n) {
                candidate = n;

                result.ep = candidate->getTraversalOrder();
                result.ex = exid;
                result.top = instanceNode->getTraversalOrder();

                break;
            }
        }

        if (result.ep < 0) {
            while (!stack[0].empty()) {
                auto c = stack[0].top();
                stack[0].pop();

                bvhnode* node = c.node;

                candidate = node;

                auto n = traverse(node, f);

                if (n) {
                    candidate = n;

                    result.ep = candidate->getTraversalOrder();
                    result.ex = 0;
                    result.top = -1;

                    break;
                }
            }
        }

        return std::move(result);
    }

    bvhnode* bvh::traverse(
        bvhnode* root,
        const frustum& f)
    {
        bvhnode* stack[32];
        int stackpos = 0;

        stack[0] = root;
        stackpos = 1;

        while (stackpos > 0) {
            bvhnode* node = stack[stackpos - 1];
            stackpos--;

            if (node->isLeaf()) {
                auto i = f.intersect(node->getBoundingbox());

                if (i) {
                    return node;
                }
            }
            else {
                auto i = f.intersect(node->getBoundingbox());

                if (i) {
                    auto left = node->getLeft();
                    auto right = node->getRight();

                    if (left && !left->isCandidate()) {
                        stack[stackpos++] = left;
                    }
                    if (right && !right->isCandidate()) {
                        stack[stackpos++] = right;
                    }
                }
            }
        }

        return nullptr;
    }

    bool bvh::findCandidates(
        bvhnode* node,
        bvhnode* instanceNode,
        const frustum& f,
        std::stack<Candidate>* stack)
    {
        if (!node) {
            return false;
        }

        node->setIsCandidate(false);

        if (node->isLeaf()) {
            auto original = node;
            bvh* nestedBvh = nullptr;
            aten::mat4 mtxW2L;

            // Check if node has nested bvh.
            {
                node = getNestedNode(original, &mtxW2L);

                if (node != original) {
                    // Register nested bvh.
                    aten::hitable* originalItem = original->getItem();
                    AT_ASSERT(originalItem->isInstance());

                    // Register relation between instance and nested bvh.
                    auto internalItem = const_cast<hitable*>(originalItem->getHasObject());

                    // TODO
                    auto obj = (AT_NAME::object*)internalItem;

                    nestedBvh = (bvh*)obj->getInternalAccelerator();

                    mtxW2L.invert();
                }

                node = original;
            }

            auto i = f.intersect(node->getBoundingbox());

            if (i) {
                if (nestedBvh) {
                    // Node has nested bvh.
                    auto transformedFrustum = f;
                    transformedFrustum.transform(mtxW2L);

                    bool found = findCandidates(
                        nestedBvh->m_root,
                        original,
                        transformedFrustum,
                        stack);
                }
                else {
                    node->setIsCandidate(true);

                    stack[instanceNode != nullptr].push(Candidate(node, instanceNode));
                }

                return true;
            }
        }

        // NOTE
        // leftからtraverseするので、もし、外した時にtraverseが続くようにするには、leftが採用されやすくならないといけない.
        // そこで、スタックの最後にleftが積まれやすくなるように先にrightから探索する.
        auto s0 = findCandidates(node->getRight(), instanceNode, f, stack);
        auto s1 = findCandidates(node->getLeft(), instanceNode, f, stack);

        if (s0 || s1) {
            node->setIsCandidate(true);
            stack[instanceNode != nullptr].push(Candidate(node, instanceNode));
        }

        return s0 || s1;
    }
}
