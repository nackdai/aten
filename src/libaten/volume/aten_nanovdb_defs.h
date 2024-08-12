#pragma once

#ifndef NANOVDB_NANOVDB_H_HAS_BEEN_INCLUDED
namespace nanovdb {
    class Coord;
    class Mask;

    template <class BuildT, class Coord, class Mask, int N> class LeafNode;
    template <class Child, int N> class InternalNode;
    template <class Upper> class RootNode;
    template <class Root> class Tree;
    template <class Tree> class Grid;

    template<typename BuildT>
    using NanoLeaf = LeafNode<BuildT, Coord, Mask, 3>;

    template<typename BuildT>
    using NanoLower = InternalNode<NanoLeaf<BuildT>, 4>;

    template<typename BuildT>
    using NanoUpper = InternalNode<NanoLower<BuildT>, 5>;

    template<typename BuildT>
    using NanoRoot = RootNode<NanoUpper<BuildT>>;

    template<typename BuildT>
    using NanoTree = Tree<NanoRoot<BuildT>>;

    using FloatTree = NanoTree<float>;
    using FloatGrid = Grid<FloatTree>;
}
#endif
