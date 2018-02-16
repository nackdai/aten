#include "MdlExporter.h"
#include "GeometryChunk.h"
#include "JointChunk.h"
#include "FileOutputStream.h"

// フォーマット
// +--------------------+
// |   ファイルヘッダ   |
// +--------------------+
// | ジオメトリチャンク |
// +--------------------+
// | ジョイントチャンク |
// +--------------------+
// | マテリアルチャンク |
// +--------------------+

bool MdlExporter::export(
    uint32_t maxJointMtxNum,
    const char* lpszOutFile,
	aten::FbxImporter* pImporter)
{
    bool ret = true;

	FileOutputStream out;

    AT_VRETURN(out.open(lpszOutFile));

    aten::S_MDL_HEADER sHeader;
    {
        FILL_ZERO(&sHeader, sizeof(sHeader));

        sHeader.sizeHeader = sizeof(sHeader);
    }

    // Blank for file's header.
    IoStreamSeekHelper seekHelper(&out);
    VGOTO(ret = seekHelper.skip(sizeof(sHeader)), __EXIT__);

#if 0
#if 1
    // geometry chunk
    ret = CGeometryChunk::getInstance().Export(
            &out,
            pImporter);
    CGeometryChunk::getInstance().Clear();
    VGOTO(ret, __EXIT__);
#endif

#if 1
    // joint chunk
    ret = CJointChunk::Export(
            &out,
            pImporter);
    VGOTO(ret, __EXIT__);
#endif

    // Export terminater.
    {
        aten::S_MDL_CHUNK_HEADER sChunkHeader;
        sChunkHeader.magicChunk = aten::MDL_CHUNK_MAGIC_TERMINATE;
        OUTPUT_WRITE_VRETURN(&out, &sChunkHeader, 0, sizeof(sChunkHeader));
    }

    // Export files's header.
    {
        sHeader.sizeFile = out.getCurPos();

        const aten::math::SVector4& vMin = CGeometryChunk::getInstance().GetMin();
        const aten::math::SVector4& vMax = CGeometryChunk::getInstance().GetMax();

        sHeader.minVtx[0] = vMin.x;
        sHeader.minVtx[1] = vMin.y;
        sHeader.minVtx[2] = vMin.z;

        sHeader.maxVtx[0] = vMax.x;
        sHeader.maxVtx[1] = vMax.y;
        sHeader.maxVtx[2] = vMax.z;

        AT_VRETURN(seekHelper.returnTo());
        OUTPUT_WRITE_VRETURN(&out, &sHeader, 0, sizeof(sHeader));
    }
#else
    // Mesh chunk.
    {
        aten::S_MDL_CHUNK_HEADER sChunkHeader;
        sChunkHeader.magicChunk = aten::MDL_CHUNK_MAGIC_MESH;
        ret = OUTPUT_WRITE(&out, &sChunkHeader, 0, sizeof(sChunkHeader));
        VGOTO(ret, __EXIT__);

        ret = CGeometryChunk::getInstance().Export(
                maxJointMtxNum,
                &out,
                pImporter);
    }

    // Skeleton chunk.
    {
        aten::S_MDL_CHUNK_HEADER sChunkHeader;
        sChunkHeader.magicChunk = aten::MDL_CHUNK_MAGIC_SKELETON;
        ret = OUTPUT_WRITE(&out, &sChunkHeader, 0, sizeof(sChunkHeader));
        VGOTO(ret, __EXIT__);

        ret = JointChunk::export(
                &out,
                pImporter);
    }

    // Export terminater.
    {
        aten::S_MDL_CHUNK_HEADER sChunkHeader;
        sChunkHeader.magicChunk = aten::MDL_CHUNK_MAGIC_TERMINATE;
        ret = OUTPUT_WRITE(&out, &sChunkHeader, 0, sizeof(sChunkHeader));
        VGOTO(ret, __EXIT__);
    }

    // Export files's header.
    {
        sHeader.sizeFile = out.getCurPos();

        VGOTO(ret = seekHelper.returnTo(), __EXIT__);
        ret = OUTPUT_WRITE(&out, &sHeader, 0, sizeof(sHeader));
        VGOTO(ret, __EXIT__);
    }
#endif

__EXIT__:
    out.finalize();

    return ret;
}
