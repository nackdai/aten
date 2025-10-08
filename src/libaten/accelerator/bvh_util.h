#pragma once

#include <functional>
#include <map>
#include <vector>

#include "accelerator/bvh.h"
#include "accelerator/bvh_node.h"
#include "scene/hitable.h"

namespace aten {
    /**
     * @brief Return the root of the nested tree which the specified node has.
     */
    inline bvhnode* getNestedNode(bvhnode* node)
    {
        bvhnode* ret = nullptr;

        if (node) {
            auto item = node->getItem();
            if (item) {
                auto internalObj = const_cast<hitable*>(item->getHasObject());
                if (internalObj) {
                    auto accel = internalObj->getInternalAccelerator();

                    // NOTE
                    // 本来ならこのキャストは不正だが、BVHであることは自明なので.
                    auto bvh = *reinterpret_cast<aten::bvh**>(&accel);

                    ret = bvh->GetRoot();
                }
            }

            if (!ret) {
                ret = node;
            }
        }

        return ret;
    }

    /**
     * @brief Convert the tree to the linear list.
     */
    template <class _T>
    void registerBvhNodeToLinearListRecursively(
        aten::bvhnode* root,
        aten::bvhnode* parentNode,
        aten::hitable* nestParent,
        const aten::mat4& mtx_L2W,
        std::vector<_T>& listBvhNode,
        std::vector<aten::accelerator*>& listBvh,
        std::map<hitable*, aten::accelerator*>& nestedBvhMap,
        std::function<void(std::vector<_T>&, aten::bvhnode*, aten::hitable*, const aten::mat4&)> funcRegisterToList,
        std::function<void(aten::bvhnode*, int32_t, int32_t)> funcIfInstanceNode)
    {
        if (!root) {
            return;
        }

        auto pnode = root;

        auto original = pnode;

        // ネストしている場合にはネストさきのツリーのルートノードを取得.
        // ネストしていない場合は同じものが返ってくる.
        pnode = getNestedNode(original);

        if (pnode != original) {
            // ネストしている.
            {
                original->setParent(parentNode);
                original->setTraversalOrder((int32_t)listBvhNode.size());
                funcRegisterToList(listBvhNode, original, nestParent, mtx_L2W);
            }

            // Register nested bvh.
            aten::hitable* originalItem = original->getItem();
            AT_ASSERT(originalItem->isInstance());

            // Register relation between instance and nested bvh.
            auto internalItem = const_cast<hitable*>(originalItem->getHasObject());
            auto internalSecondItem = const_cast<hitable*>(originalItem->getHasSecondObject());

            hitable* items[] = {
                internalItem,
                internalSecondItem,
            };

            int32_t exids[2] = { -1, -1 };

            for (int32_t i = 0; i < AT_COUNTOF(items); i++) {
                auto item = items[i];

                if (item == nullptr) {
                    break;
                }

                // TODO
                auto obj = (AT_NAME::PolygonObject*)item;

                auto nestedBvh = obj->getInternalAccelerator();

                auto found = std::find(listBvh.begin(), listBvh.end(), nestedBvh);
                if (found == listBvh.end()) {
                    listBvh.push_back(nestedBvh);

                    int32_t exid = (int32_t)listBvh.size() - 1;
                    AT_ASSERT(exid >= 0);

                    exids[i] = exid;
                }

                if (i == 0) {
                    if (nestedBvh) {
                        nestedBvhMap.insert(std::pair<aten::hitable*, aten::accelerator*>(originalItem, nestedBvh));
                    }
                }
            }

            if (funcIfInstanceNode) {
                funcIfInstanceNode(original, exids[0], exids[1]);
            }
        }
        else {
            pnode->setParent(parentNode);
            pnode->setTraversalOrder((int32_t)listBvhNode.size());
            funcRegisterToList(listBvhNode, pnode, nestParent, mtx_L2W);

            aten::bvhnode* pleft = pnode->getLeft();
            aten::bvhnode* pright = pnode->getRight();

            registerBvhNodeToLinearListRecursively(
                pleft, pnode, nestParent, mtx_L2W, listBvhNode, listBvh, nestedBvhMap, funcRegisterToList, funcIfInstanceNode);
            registerBvhNodeToLinearListRecursively(
                pright, pnode, nestParent, mtx_L2W, listBvhNode, listBvh, nestedBvhMap, funcRegisterToList, funcIfInstanceNode);
        }
    }
}
