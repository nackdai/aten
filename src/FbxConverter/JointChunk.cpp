#include "JointChunk.h"
#include "GeometryCommon.h"
#include "FileOutputStream.h"

#include "math/quaternion.h"
#include "misc/key.h"

// ジョイントチャンク
// +----------------------+
// |    チャンクヘッダ    |
// +----------------------+
// |  ジョイントチャンク  |
// |       ヘッダ         |
// +----------------------+
// |  ジョイントテーブル  |
// | +------------------+ |
// | |     ジョイント   | |
// | |+----------------+| |
// | ||     ヘッダ     || |
// | |+----------------+| |
// | |+----------------+| |
// | ||    変形データ  || |
// | |+----------------+| |
// | |       ・・・     | |
// | +------------------+ |
// |        ・・・        |
// +----------------------+

bool JointChunk::export(
	FileOutputStream* pOut,
	aten::FbxImporter* pImporter)
{
	AT_VRETURN(pImporter->beginJoint());

	uint32_t nJointNum = pImporter->getJointNum();
	if (nJointNum == 0) {
		// There is no joints, so this function fihish.
		return true;
	}

	aten::JointHeader sHeader;
	{
		// TODO
		// version, magic number

		sHeader.numJoint = nJointNum;
		sHeader.sizeHeader = sizeof(sHeader);
	}

	// Blank for S_SKL_HEADER.
	IoStreamSeekHelper seekHelper(pOut);
	AT_VRETURN(seekHelper.skip(sizeof(sHeader)));

	std::vector<aten::JointParam> tvJoint;

	tvJoint.resize(pImporter->getJointNum());

	sHeader.numJoint = (uint32_t)tvJoint.size();

	// Get joint's info.
	getJointInfo(
		pOut,
		pImporter,
		tvJoint);

	// Export.
	AT_VRETURN(
		exportJoint(
			pOut,
			pImporter,
			tvJoint));

	pImporter->endJoint();

	// Export S_SKL_HEADER.
	{
		sHeader.sizeFile = pOut->getCurPos();

		// Rmenber end of geometry chunk.
		AT_VRETURN(seekHelper.returnWithAnchor());

		OUTPUT_WRITE_VRETURN(pOut, &sHeader, 0, sizeof(sHeader));
		seekHelper.step(sizeof(sHeader));

		// returnTo end of geometry chunk.
		AT_VRETURN(seekHelper.returnToAnchor());
	}

	pImporter->exportJointCompleted();

	return true;
}

void JointChunk::getJointInfo(
	FileOutputStream* pOut,
	aten::FbxImporter* pImporter,
	std::vector<aten::JointParam>& tvJoint)
{
	for (uint32_t i = 0; i < (uint32_t)tvJoint.size(); i++) {
		auto& sJoint = tvJoint[i];

		sJoint.name.SetString(
			pImporter->getJointName(i));
		sJoint.nameKey = aten::Key::gen(sJoint.name);

		sJoint.idx = i;

		pImporter->getJointInvMtx(
			i,
			sJoint.mtxInvBind);
	}

	for (size_t i = 0; i < tvJoint.size(); i++) {
		auto& sJoint = tvJoint[i];

		sJoint.parent = pImporter->getJointParent((uint32_t)i, tvJoint);
	}
}

bool JointChunk::exportJoint(
	FileOutputStream* pOut,
	aten::FbxImporter* pImporter,
	std::vector<aten::JointParam>& tvJoint)
{
	for (size_t i = 0; i < tvJoint.size(); i++) {
		auto& sJoint = tvJoint[i];

		std::vector<JointTransformParam> tvTransform;

		pImporter->getJointTransform(
			(uint32_t)i,
			tvJoint,
			tvTransform);

		aten::mat4 mtxRot;

		sJoint.validParam = 0;
		sJoint.validAnmParam = 0;

		sJoint.pose.trans[0] = 0.0f;
		sJoint.pose.trans[1] = 0.0f;
		sJoint.pose.trans[2] = 0.0f;

		sJoint.pose.scale[0] = 1.0f;
		sJoint.pose.scale[1] = 1.0f;
		sJoint.pose.scale[2] = 1.0f;

		sJoint.pose.quat.identity();

		bool bHasQuatFromAxisRot = false;

		for (size_t n = 0; n < tvTransform.size(); n++) {
			JointTransformParam& sTransform = tvTransform[n];

			AT_ASSERT(sTransform.param.size() > 0);

			if (sTransform.type == JointTransform::Translate) {
				AT_ASSERT(sTransform.param.size() >= 3);
				sJoint.pose.trans[0] = sTransform.param[0];
				sJoint.pose.trans[1] = sTransform.param[1];
				sJoint.pose.trans[2] = sTransform.param[2];

				sJoint.validParam |= aten::JointTransformType::Translate;
			}
			else if (sTransform.type == JointTransform::Scale) {
				AT_ASSERT(sTransform.param.size() >= 3);
				sJoint.pose.scale[0] = sTransform.param[0];
				sJoint.pose.scale[1] = sTransform.param[1];
				sJoint.pose.scale[2] = sTransform.param[2];

				sJoint.validParam |= aten::JointTransformType::Scale;
			}
			else if (sTransform.type == JointTransform::AxisRot) {
				float x = sTransform.param[0];
				float y = sTransform.param[1];
				float z = sTransform.param[2];
				float angle = sTransform.param[3];

				aten::quat quat;
				quat.setQuatFromRadAxis(angle, aten::vec4(x, y, z, real(0)));

				auto mtx = quat.getMatrix();

				mtxRot = mtx * mtxRot;

				bHasQuatFromAxisRot = true;
			}
			else if (sTransform.type == JointTransform::Quaternion) {
				AT_ASSERT(sTransform.param.size() == 4);
				sJoint.pose.quat.x = sTransform.param[0];
				sJoint.pose.quat.y = sTransform.param[1];
				sJoint.pose.quat.z = sTransform.param[2];
				sJoint.pose.quat.w = sTransform.param[3];

				sJoint.validParam |= aten::JointTransformType::Quaternion;
			}
			else {
				AT_VRETURN(false);
			}
		}

		if (bHasQuatFromAxisRot) {
			aten::quat quat;
			quat.fromMatrix(mtxRot);

			sJoint.pose.quat = quat;

			sJoint.validParam |= aten::JointTransformType::Quaternion;
		}

		OUTPUT_WRITE_VRETURN(pOut, &sJoint, 0, sizeof(sJoint));
	}

	return true;
}
