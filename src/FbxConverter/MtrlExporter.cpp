#include "MtrlExporter.h"
#include "FileOutputStream.h"

#if 0
bool MtrlExporter::exportMaterial(
    const char* lpszOutFile,
    aten::FbxImporter* pImporter,
    uint32_t nMtrlIdx)
{
    bool ret = true;

	FileOutputStream out;

    AT_VRETURN(pImporter->beginMaterial());

    uint32_t nMtrlNum = pImporter->getMaterialNum();
    AT_ASSERT(nMtrlIdx < nMtrlNum);

    // Open file.
    AT_VRETURN(out.open(lpszOutFile));

    aten::S_MTRL_HEADER sHeader;
    {
        FILL_ZERO(&sHeader, sizeof(sHeader));

        sHeader.sizeHeader = sizeof(sHeader);

        // TODO
        // magic number, version...
    }

    // Blank for file's header and jump table.
    IoStreamSeekHelper seekHelper(&out);
#if 0
    AT_VRETURN(seekHelper.skip(sizeof(sHeader) + sizeof(uint32_t) * nMtrlNum));

    // Jump table
    std::vector<uint32_t> tvJumpTbl;

    for (uint32_t i = 0; i < nMtrlNum; i++) {
        // Add position to jump table.,
        tvJumpTbl.push_back(m_Out.getCurPos());

        AT_VRETURN(ExportMaterial(i, pImporter));
    }
#else
    AT_VRETURN(seekHelper.skip(sizeof(sHeader)));
    AT_VRETURN(exportMaterial(out, nMtrlIdx, pImporter));
#endif

    AT_VRETURN(pImporter->endMaterial());

    // Export files's header and jump table.
    {
#if 0
        sHeader.numMtrl = nMtrlNum;
#endif

        sHeader.sizeFile = out.getCurPos();

        // returnTo to file's top.
        AT_VRETURN(seekHelper.returnTo());

        // Export files' header.
        OUTPUT_WRITE_VRETURN(&out, &sHeader, 0, sizeof(sHeader));

#if 0
        // Export jump table.
        OUTPUT_WRITE_VRETURN(&out, &tvJumpTbl[0], 0, sizeof(uint32_t) * nMtrlNum);
#endif
    }

	out.finalize();

    return true;
}

bool MtrlExporter::exportMaterial(
	FileOutputStream& out,
    uint32_t nMtrlIdx,
	aten::FbxImporter* pImporter)
{
    aten::S_MTRL_MATERIAL sMtrl;
    FILL_ZERO(&sMtrl, sizeof(sMtrl));

    AT_VRETURN(
        pImporter->getMaterial(
            nMtrlIdx,
            sMtrl));

    OUTPUT_WRITE_VRETURN(&out, &sMtrl, 0, sizeof(sMtrl));

    // Export textrure's info.
    for (uint32_t i = 0; i < sMtrl.numTex; i++) {
        aten::S_MTRL_TEXTURE sTex;
        FILL_ZERO(&sTex, sizeof(sTex));

        sTex.idx = i;

        pImporter->getMaterialTexture(
            nMtrlIdx,
            i,
            sTex);

        OUTPUT_WRITE_VRETURN(&out, &sTex, 0, sizeof(sTex));
    }

    // Export shader's info.
    //for (uint32_t i = 0; i < sMtrl.numShader; i++) {

    // NOTE
    // Material has only one shader.
    for (uint32_t i = 0; i < 1; i++) {
        aten::S_MTRL_SHADER sShader;
        FILL_ZERO(&sShader, sizeof(sShader));

        pImporter->getMaterialShader(
            nMtrlIdx,
            i,
            sShader);

        OUTPUT_WRITE_VRETURN(&m_Out, &sShader, 0, sizeof(sShader));
    }

    std::vector<aten::S_MTRL_PARAM> tvParam(sMtrl.numParam);

    // Export parameter's info.
    for (uint32_t i = 0; i < sMtrl.numParam; i++) {
        aten::S_MTRL_PARAM& sParam = tvParam[i];
        FILL_ZERO(&sParam, sizeof(sParam));

        pImporter->getMaterialParam(
            nMtrlIdx,
            i,
            sParam);

        sParam.idx = i;

        OUTPUT_WRITE_VRETURN(&out, &sParam, 0, sizeof(sParam));
    }

    // Export parameter's value.
    for (uint32_t i = 0; i < sMtrl.numParam; i++) {
        const aten::S_MTRL_PARAM& sParam = tvParam[i];

        if (sParam.elements > 0) {
            std::vector<float> tvValue;
            pImporter->getMaterialParamValue(
                nMtrlIdx,
                i,
                tvValue);

            switch (sParam.type) {
            case aten::E_MTRL_PARAM_TYPE_FLOAT:
            case aten::E_MTRL_PARAM_TYPE_VECTOR:
            case aten::E_MTRL_PARAM_TYPE_MATRIX:
            {
                OUTPUT_WRITE_VRETURN(&m_Out, &tvValue[0], 0, sParam.bytes);
                break;
            }
            case aten::E_MTRL_PARAM_TYPE_UINT:
            case aten::E_MTRL_PARAM_TYPE_BOOL:
                // TODO
                AT_ASSERT(false);
                break;
            }
        }
    }

    return true;
}
#endif
