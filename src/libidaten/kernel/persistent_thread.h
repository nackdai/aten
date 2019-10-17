#pragma once

// NOTE
// persistent thread.
// https://gist.github.com/guozhou/b972bb42bbc5cba1f062#file-persistent-cpp-L15

// NOTE
// compute capability 7.5
// https://en.wikipedia.org/wiki/CUDA

#define NUM_SM				30    // no. of streaming multiprocessors
#define NUM_WARP_PER_SM     32    // maximum no. of resident warps per SM
#define NUM_BLOCK_PER_SM    16    // maximum no. of resident blocks per SM
#define NUM_BLOCK           (NUM_SM * NUM_BLOCK_PER_SM)
#define NUM_WARP_PER_BLOCK  (NUM_WARP_PER_SM / NUM_BLOCK_PER_SM)
#define WARP_SIZE           32
