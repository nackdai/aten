#include "JointChunk.h"
#include "GeometryCommon.h"
#include "FileOutputStream.h"

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

	izanagi::S_SKL_HEADER sHeader;
	{
		FILL_ZERO(&sHeader, sizeof(sHeader));

		// TODO
		// version, magic number

		sHeader.numJoint = nJointNum;
		sHeader.sizeHeader = sizeof(sHeader);
	}

	// Blank for S_SKL_HEADER.
	IoStreamSeekHelper seekHelper(pOut);
	AT_VRETURN(seekHelper.skip(sizeof(sHeader)));

	std::vector<izanagi::S_SKL_JOINT> tvJoint;

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
	std::vector<izanagi::S_SKL_JOINT>& tvJoint)
{
	for (uint32_t i = 0; i < (uint32_t)tvJoint.size(); i++) {
		izanagi::S_SKL_JOINT& sJoint = tvJoint[i];

		sJoint.name.SetString(
			pImporter->GetJointName(i));
		sJoint.nameKey = sJoint.name.GetKeyValue();

		sJoint.idx = i;

		pImporter->getJointInvMtx(
			i,
			sJoint.mtxInvBind);
	}

	for (size_t i = 0; i < tvJoint.size(); i++) {
		izanagi::S_SKL_JOINT& sJoint = tvJoint[i];

		sJoint.parent = pImporter->getJointParent((uint32_t)i, tvJoint);
	}
}

bool JointChunk::ExportJoint(
	FileOutputStream* pOut,
	aten::FbxImporter* pImporter,
	std::vector<izanagi::S_SKL_JOINT>& tvJoint)
{
	for (size_t i = 0; i < tvJoint.size(); i++) {
		izanagi::S_SKL_JOINT& sJoint = tvJoint[i];

		std::vector<JointTransformParam> tvTransform;

		pImporter->GetJointTransform(
			(uint32_t)i,
			tvJoint,
			tvTransform);

		izanagi::math::SMatrix44 mtxRot;
		izanagi::math::SMatrix44::SetUnit(mtxRot);

		sJoint.validParam = 0;
		sJoint.validAnmParam = 0;

		sJoint.pose.trans[0] = 0.0f;
		sJoint.pose.trans[1] = 0.0f;
		sJoint.pose.trans[2] = 0.0f;

		sJoint.pose.scale[0] = 1.0f;
		sJoint.pose.scale[1] = 1.0f;
		sJoint.pose.scale[2] = 1.0f;

		sJoint.pose.quat.Set(0.0f, 0.0f, 0.0f, 1.0f);

		bool bHasQuatFromAxisRot = false;

		for (size_t n = 0; n < tvTransform.size(); n++) {
			JointTransformParam& sTransform = tvTransform[n];

			AT_ASSERT(sTransform.param.size() > 0);

			if (sTransform.type == JointTransform::Translate) {
				AT_ASSERT(sTransform.param.size() >= 3);
				sJoint.pose.trans[0] = sTransform.param[0];
				sJoint.pose.trans[1] = sTransform.param[1];
				sJoint.pose.trans[2] = sTransform.param[2];

				sJoint.validParam |= izanagi::E_SKL_JOINT_PARAM_TRANSLATE;
			}
			else if (sTransform.type == JointTransform::Scale) {
				AT_ASSERT(sTransform.param.size() >= 3);
				sJoint.pose.scale[0] = sTransform.param[0];
				sJoint.pose.scale[1] = sTransform.param[1];
				sJoint.pose.scale[2] = sTransform.param[2];

				sJoint.validParam |= izanagi::E_SKL_JOINT_PARAM_SCALE;
			}
			else if (sTransform.type == JointTransform::AxisRot) {
				float x = sTransform.param[0];
				float y = sTransform.param[1];
				float z = sTransform.param[2];
				float angle = sTransform.param[3];

				izanagi::math::SQuat quat;
				izanagi::math::SQuat::SetQuatFromRadAxis(quat, angle, x, y, z);

				izanagi::math::SMatrix44 mtx;
				izanagi::math::SQuat::MatrixFromQuat(mtx, quat);

				izanagi::math::SMatrix44::Mul(mtxRot, mtxRot, mtx);

				bHasQuatFromAxisRot = true;
			}
			else if (sTransform.type == JointTransform::Quaternion) {
				AT_ASSERT(sTransform.param.size() == 4);
				sJoint.pose.quat.x = sTransform.param[0];
				sJoint.pose.quat.y = sTransform.param[1];
				sJoint.pose.quat.z = sTransform.param[2];
				sJoint.pose.quat.w = sTransform.param[3];

				sJoint.validParam |= izanagi::E_SKL_JOINT_PARAM_QUARTANION;
			}
			else {
				AT_VRETURN(false);
			}
		}

		if (bHasQuatFromAxisRot) {
			izanagi::math::SQuat quat;
			izanagi::math::SQuat::QuatFromMatrix(quat, mtxRot);

			izanagi::math::SQuat::Copy(sJoint.pose.quat, quat);

			sJoint.validParam |= izanagi::E_SKL_JOINT_PARAM_QUARTANION;
		}

		OUTPUT_WRITE_VRETURN(pOut, &sJoint, 0, sizeof(sJoint));
	}

	return true;
}
