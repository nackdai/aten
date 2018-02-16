#include "AnmExporter.h"
#include "FileOutputStream.h"

#include "deformable/ANMFormat.h"

// NOTE
// フォーマット
// +----------------+
// | ファイルヘッダ |
// +----------------+
// | アニメーション |
// |     ノード     |
// +----------------+
// | アニメーション |
// |   チャンネル   |
// +----------------+
// | アニメーション |
// |      キー      |
// +----------------+

bool AnmExporter::export(
    const char* lpszOutFile,
    uint32_t nSetIdx,
    aten::FbxImporter* pImporter)
{
    bool ret = true;

	FileOutputStream out;

    AT_VRETURN(out.open(lpszOutFile));

    aten::AnmHeader sHeader;
    {
        sHeader.sizeHeader = sizeof(sHeader);

        // TODO
		sHeader.keyType = aten::AnmKeyType::Time;
    }

    // Blank for file's header.
    IoStreamSeekHelper seekHelper(&out);
    AT_VRETURN(seekHelper.skip(sizeof(sHeader)));

    AT_VRETURN(pImporter->beginAnm(nSetIdx));

    uint32_t nNodeNum = pImporter->getAnmNodeNum();

    std::vector<uint32_t> channelNum;

    // Export nodes.
    {
        uint32_t channelIdx = 0;

        for (uint32_t i = 0; i < nNodeNum; i++) {
            aten::AnmNode sNode;

            AT_VRETURN(pImporter->getAnmNode(i, sNode));

            channelNum.push_back(sNode.numChannels);

            sNode.channelIdx = channelIdx;
            channelIdx += sNode.numChannels;

            OUTPUT_WRITE_VRETURN(&out, &sNode, 0, sizeof(sNode));
        }
    }

    uint32_t nChannelNum = 0;
    std::vector<uint32_t> tvKeyNum;

    uint32_t keyIdx = 0;

    // Export channels,
    for (uint32_t nNodeIdx = 0; nNodeIdx < nNodeNum; nNodeIdx++) {
        //uint32_t nChannelCnt = pImporter->GetAnmChannelNum(nNodeIdx);
        uint32_t nChannelCnt = channelNum[nNodeIdx];

        for (uint32_t nChannelIdx = 0; nChannelIdx < nChannelCnt; nChannelIdx++) {
            aten::AnmChannel sChannel;

            AT_VRETURN(
                pImporter->getAnmChannel(
                    nNodeIdx,
                    nChannelIdx,
                    sChannel));

            sChannel.keyIdx = keyIdx;
            keyIdx += sChannel.numKeys;

            OUTPUT_WRITE_VRETURN(&out, &sChannel, 0, sizeof(sChannel));

            tvKeyNum.push_back(sChannel.numKeys);
        }

        nChannelNum += nChannelCnt;
    }

    uint32_t nKeyNum = 0;
    uint32_t nKeyPos = 0;

    float fMaxTime = 0.0f;

    // Export keys.
    for (uint32_t nNodeIdx = 0; nNodeIdx < nNodeNum; nNodeIdx++) {
        //uint32_t nChannelCnt = pImporter->GetAnmChannelNum(nNodeIdx);
        uint32_t nChannelCnt = channelNum[nNodeIdx];

        for (uint32_t nChannelIdx = 0; nChannelIdx < nChannelCnt; nChannelIdx++) {
            uint32_t nKeyCnt = tvKeyNum[nKeyPos++];

            for (uint32_t nKeyIdx = 0; nKeyIdx < nKeyCnt; nKeyIdx++) {
                aten::AnmKey sKey;

                std::vector<float> tvValue;

                AT_VRETURN(
                    pImporter->getAnmKey(
                        nNodeIdx,
                        nChannelIdx,
                        nKeyIdx,
                        sKey,
                        tvValue));

                fMaxTime = (sKey.keyTime > fMaxTime ? sKey.keyTime : fMaxTime);

                sKey.numParams = static_cast<uint8_t>(tvValue.size());
                sKey.value = 0;

                OUTPUT_WRITE_VRETURN(&out, &sKey, 0, sizeof(sKey));

#if 0
                int32_t nOffset = sizeof(float);
                AT_VRETURN(out.Seek(-nOffset, aten::E_IO_STREAM_SEEK_POS_CUR));
#endif

                OUTPUT_WRITE_VRETURN(&out, &tvValue[0], 0, sizeof(float) * tvValue.size());
            }

            nKeyNum += nKeyCnt;
        }
    }

    AT_VRETURN(pImporter->endAnm());

    // Export files's header.
    {
        sHeader.numNodes = nNodeNum;
        sHeader.numChannels = nChannelNum;
        sHeader.numKeys = nKeyNum;

        sHeader.sizeFile = out.getCurPos();

        sHeader.time = fMaxTime;

        AT_VRETURN(seekHelper.returnTo());
        OUTPUT_WRITE_VRETURN(&out, &sHeader, 0, sizeof(sHeader));
    }

    return true;
}
