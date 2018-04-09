#include "MdlExporter.h"
#include "GeometryChunk.h"
#include "JointChunk.h"
#include "FileOutputStream.h"

#include "deformable/MDLFormat.h"

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

#define VGOTO(b, label) if (!(b)) { goto label; }

bool MdlExporter::exportMdl(
    uint32_t maxJointMtxNum,
    const char* lpszOutFile,
	aten::FbxImporter* pImporter,
	bool isExportForGPUSkinning/*= false*/)
{
    bool ret = true;

	FileOutputStream out;

    AT_VRETURN_FALSE(out.open(lpszOutFile));

    aten::MdlHeader sHeader;
    {
        sHeader.sizeHeader = sizeof(sHeader);
    }

    // Blank for file's header.
    IoStreamSeekHelper seekHelper(&out);
    VGOTO(ret = seekHelper.skip(sizeof(sHeader)), __EXIT__);

    // Mesh chunk.
    {
        aten::MdlChunkHeader sChunkHeader;
        sChunkHeader.magicChunk = aten::MdlChunkMagic::Mesh;
        ret = OUTPUT_WRITE(&out, &sChunkHeader, 0, sizeof(sChunkHeader));
        VGOTO(ret, __EXIT__);

        ret = GeometryChunkExporter::getInstance().exportGeometry(
			maxJointMtxNum,
            &out,
            pImporter,
			isExportForGPUSkinning);
    }

    // Skeleton chunk.
    {
        aten::MdlChunkHeader sChunkHeader;
        sChunkHeader.magicChunk = aten::MdlChunkMagic::Joint;
        ret = OUTPUT_WRITE(&out, &sChunkHeader, 0, sizeof(sChunkHeader));
        VGOTO(ret, __EXIT__);

        ret = JointChunk::exportJoint(
			&out,
            pImporter);
    }

    // Export terminater.
    {
        aten::MdlChunkHeader sChunkHeader;
        sChunkHeader.magicChunk = aten::MdlChunkMagic::Terminate;
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

__EXIT__:
    out.finalize();

    return ret;
}
