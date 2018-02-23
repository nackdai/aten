#include "FbxImporter.h"
#include "FbxDataManager.h"

#include "misc/key.h"

#include <map>
#include <algorithm>
#include <iterator>
#include <functional>

// NOTE
// http://www.gamedev.net/page/resources/_/technical/graphics-programming-and-theory/how-to-work-with-fbx-sdk-r3582

namespace aten
{
	FbxImporter::FbxImporter()
	{
	}

	bool FbxImporter::open(const char* pszName, bool isOpenForAnm)
	{
		bool ret = false;
		m_dataMgr = new FbxDataManager();
    
		if (isOpenForAnm) {
			ret = m_dataMgr->openForAnm(pszName);
		}
		else {
			ret = m_dataMgr->open(pszName);
		}

		return ret;
	}

	bool FbxImporter::close()
	{
		if (m_dataMgr) {
			m_dataMgr->close();
			delete m_dataMgr;
		}

		if (m_dataMgrBase) {
			m_dataMgrBase->close();
			delete m_dataMgrBase;
		}

		return true;
	}

	//////////////////////////////////
	// For geometry chunk.

	void FbxImporter::exportGeometryCompleted()
	{
		m_posVtx = 0;
	}

	void FbxImporter::beginMesh(uint32_t nIdx)
	{
		m_curMeshIdx = nIdx;
	}

	void FbxImporter::endMesh()
	{
	}

	uint32_t FbxImporter::getMeshNum()
	{
		m_dataMgr->loadMesh();
		uint32_t ret = m_dataMgr->getMeshNum();
		return ret;
	}

	// スキニング情報を取得.
	void FbxImporter::getSkinList(std::vector<SkinParam>& tvSkinList)
	{
		if (tvSkinList.size() > 0) {
			return;
		}

		uint32_t vtxNum = m_dataMgr->getVtxNum();

		tvSkinList.resize(vtxNum);

		for (uint32_t i = 0; i < vtxNum; i++)
		{
			std::vector<float> weight;
			std::vector<uint32_t> joint;

			m_dataMgr->getSkinData(i, weight, joint);

			AT_ASSERT(weight.size() == joint.size());

			for (uint32_t n = 0; n < weight.size(); n++)
			{
				tvSkinList[i].add(joint[n], weight[n]);
			}
		}
	}

	// 指定されているメッシュに含まれる三角形を取得.
	uint32_t FbxImporter::getTriangles(std::vector<TriangleParam>& tvTriList)
	{
		const MeshSubset& mesh = m_dataMgr->getMesh(m_curMeshIdx);

		for each (const Face& face in mesh.faces)
		{
			TriangleParam tri;

			tri.vtx[0] = face.vtx[0];
			tri.vtx[1] = face.vtx[1];
			tri.vtx[2] = face.vtx[2];

			tvTriList.push_back(tri);
		}

		uint32_t vtxNum = mesh.vtxNum;

		return vtxNum;
	}

	uint32_t FbxImporter::getSkinIdxAffectToVtx(uint32_t nVtxIdx)
	{
		// NOTE
		// 頂点インデックスとスキン情報のインデックスは一致する.
		return nVtxIdx;
	}

	uint32_t FbxImporter::getVtxSize()
	{
		// NOTE
		// skinのサイズは外でやるのでここではやらない

		auto& meshSet = m_dataMgr->getMesh(m_curMeshIdx);

		uint32_t ret = 0;

		FbxMesh* mesh = meshSet.fbxMesh;

		// ポジションが存在しないことはないはず.
		ret += (uint32_t)aten::MeshVertexSize::Position;

		if (mesh->GetElementNormalCount() > 0) {
			ret += (uint32_t)aten::MeshVertexSize::Normal;
		}

		if (mesh->GetElementUVCount() > 0) {
			ret += (uint32_t)aten::MeshVertexSize::UV;
		}

		if (mesh->GetElementVertexColorCount() > 0) {
			ret += (uint32_t)aten::MeshVertexSize::Color;
		}

		return ret;
	}

	uint32_t FbxImporter::getVtxFmt()
	{
		auto& meshSet = m_dataMgr->getMesh(m_curMeshIdx);

		uint32_t ret = 0;

		FbxMesh* mesh = meshSet.fbxMesh;

		// ポジションが存在しないことはないはず.
		ret |= 1 << (uint32_t)aten::MeshVertexFormat::Position;

		if (mesh->GetElementNormalCount() > 0) {
			ret |= 1 << (uint32_t)aten::MeshVertexFormat::Normal;
		}

		if (mesh->GetElementUVCount() > 0) {
			ret |= 1 << (uint32_t)aten::MeshVertexFormat::UV;
		}

		if (mesh->GetElementVertexColorCount() > 0) {
			ret |= 1 << (uint32_t)aten::MeshVertexFormat::Color;
		}

		// NOTE
		// skinのフォーマットは外でやるのでここではやらない

		return ret;
	}

	static bool GetVertexData(
		fbxsdk::FbxLayerElement::EMappingMode mappingMode,
		fbxsdk::FbxLayerElement::EReferenceMode referenceMode,
		uint32_t vtxIdx, uint32_t vtxCounter,
		std::function<void(uint32_t)> funcDirect,
		std::function<void(uint32_t)> funcIndex)
	{
		bool ret = false;

		switch (mappingMode)
		{
		case FbxGeometryElement::eByControlPoint:
			switch (referenceMode)
			{
			case FbxGeometryElement::eDirect:
			{
				funcDirect(vtxIdx);
				ret = true;
			}
				break;

			case FbxGeometryElement::eIndexToDirect:
			{
				funcIndex(vtxIdx);
				ret = true;
			}
				break;

			default:
				throw std::exception("Invalid Reference");
			}
			break;

		case FbxGeometryElement::eByPolygonVertex:
			// NOTE
			// 頂点の順番でアクセスする場合.
			switch (referenceMode)
			{
			case FbxGeometryElement::eDirect:
			{
				funcDirect(vtxCounter);
				ret = true;
			}
				break;

			case FbxGeometryElement::eIndexToDirect:
			{
				funcIndex(vtxCounter);
				ret = true;
			}
				break;

			default:
				throw std::exception("Invalid Reference");
			}
			break;
		}

		return ret;
	}

	bool FbxImporter::getVertex(
		uint32_t nIdx,
		aten::vec4& vec,
		aten::MeshVertexFormat type)
	{
		const VertexData& vtx = m_dataMgr->getVertex(nIdx);

		auto mesh = m_dataMgr->getMesh(0).fbxMesh;

		if (type == aten::MeshVertexFormat::Position) {
			vec.x = static_cast<float>(vtx.pos.mData[0]);
			vec.y = static_cast<float>(vtx.pos.mData[1]);
			vec.z = static_cast<float>(vtx.pos.mData[2]);
			vec.w = 1.0f;

			return true;
		}

		if (type == aten::MeshVertexFormat::Normal) {
			if (mesh->GetElementNormalCount() > 0) {
	#if 0
				// TODO
				const FbxGeometryElementNormal* vtxNormal = mesh->GetElementNormal();

				uint32_t numDirect = vtxNormal->GetDirectArray().GetCount();
				uint32_t numIndex = vtxNormal->GetIndexArray().GetCount();

				auto mappingMode = vtxNormal->GetMappingMode();
				auto referenceMode = vtxNormal->GetReferenceMode();

				GetVertexData(
					mappingMode, referenceMode,
					realVtxIdx, countInFbxMesh,
					[&](uint32_t idx) {
					const FbxVector4& nml = vtxNormal->GetDirectArray().GetAt(idx);
					vec.x = static_cast<float>(nml.mData[0]);
					vec.y = static_cast<float>(nml.mData[1]);
					vec.z = static_cast<float>(nml.mData[2]);
				},
					[&](uint32_t idx) {
					int index = vtxNormal->GetIndexArray().GetAt(idx);
					const FbxVector4& nml = vtxNormal->GetDirectArray().GetAt(index);
					vec.x = static_cast<float>(nml.mData[0]);
					vec.y = static_cast<float>(nml.mData[1]);
					vec.z = static_cast<float>(nml.mData[2]);
				});
	#else
				vec.x = static_cast<float>(vtx.nml.mData[0]);
				vec.y = static_cast<float>(vtx.nml.mData[1]);
				vec.z = static_cast<float>(vtx.nml.mData[2]);
	#endif
				return true;
			}
		}

		if (type == aten::MeshVertexFormat::UV) {
			if (mesh->GetElementUVCount() > 0) {
				vec.x = static_cast<float>(vtx.uv.mData[0]);
				vec.y = static_cast<float>(vtx.uv.mData[1]);

				return true;
			}
		}

		if (type == aten::MeshVertexFormat::Color) {
			if (mesh->GetElementVertexColorCount() > 0) {
	#if 0
				const FbxGeometryElementVertexColor* vtxClr = mesh->GetElementVertexColor();

				uint32_t numDirect = vtxClr->GetDirectArray().GetCount();
				uint32_t numIndex = vtxClr->GetIndexArray().GetCount();

				auto mappingMode = vtxClr->GetMappingMode();
				auto referenceMode = vtxClr->GetReferenceMode();

				GetVertexData(
					mappingMode, referenceMode,
					realVtxIdx, countInFbxMesh,
					[&](uint32_t idx) {
					const FbxColor& clr = vtxClr->GetDirectArray().GetAt(idx);
					vec.x = static_cast<float>(clr.mRed);
					vec.y = static_cast<float>(clr.mGreen);
					vec.z = static_cast<float>(clr.mBlue);
					vec.w = static_cast<float>(clr.mAlpha);
				},
					[&](uint32_t idx) {
					int index = vtxClr->GetIndexArray().GetAt(idx);
					const FbxColor& clr = vtxClr->GetDirectArray().GetAt(index);
					vec.x = static_cast<float>(clr.mRed);
					vec.y = static_cast<float>(clr.mGreen);
					vec.z = static_cast<float>(clr.mBlue);
					vec.w = static_cast<float>(clr.mAlpha);
				});
	#else
				vec.x = static_cast<float>(vtx.clr.mRed);
				vec.y = static_cast<float>(vtx.clr.mGreen);
				vec.z = static_cast<float>(vtx.clr.mBlue);
				vec.w = static_cast<float>(vtx.clr.mAlpha);
	#endif       
				return true;
			}
		}

		return false;
	}

	void FbxImporter::getMaterialForMesh(
		uint32_t nMeshIdx,
		aten::MeshMaterial& sMtrl)
	{
		const MeshSubset& mesh = m_dataMgr->getMesh(nMeshIdx);

		sprintf(sMtrl.name, "%s\0", mesh.mtrl->GetName());
		sMtrl.nameKey = aten::KeyValue::gen(sMtrl.name);
	}

	//////////////////////////////////
	// For joint chunk.

	void FbxImporter::exportJointCompleted()
	{
	}

	bool FbxImporter::beginJoint()
	{
		return true;
	}

	void FbxImporter::endJoint()
	{
	}

	uint32_t FbxImporter::getJointNum()
	{
		uint32_t ret = m_dataMgr->getNodeNum();
		return ret;
	}

	const char* FbxImporter::getJointName(uint32_t nIdx)
	{
		FbxNode* node = m_dataMgr->getFbxNode(nIdx);
		return node->GetName();
	}

	int32_t FbxImporter::getJointParent(
		uint32_t nIdx,
		const std::vector<aten::JointParam>& tvJoint)
	{
		FbxNode* node = m_dataMgr->getFbxNode(nIdx);

		const FbxNode* parent = node->GetParent();
		if (parent == NULL) {
			return -1;
		}

		int32_t ret = m_dataMgr->getNodeIndex(parent);

		return ret;
	}

	void FbxImporter::getJointInvMtx(
		uint32_t nIdx,
		aten::mat4& mtx)
	{
		FbxNode* node = m_dataMgr->getFbxNode(nIdx);
		FbxCluster* cluster = m_dataMgr->getClusterByNode(node);

		if (cluster) {
	#if 0
			FbxAMatrix& mtxGlobal = node->EvaluateGlobalTransform();
			FbxAMatrix& mtxLocal = node->EvaluateLocalTransform();

			FbxVector4 trans = node->LclTranslation;
			FbxVector4 rot = node->LclRotation;
			FbxVector4 scale = node->LclScaling;

			FbxAMatrix tmp(trans, rot, scale);

			FbxAMatrix mtxTransformLink;
			cluster->GetTransformLinkMatrix(mtxTransformLink);

			FbxAMatrix mtxTransform;
			cluster->GetTransformMatrix(mtxTransform);

			FbxAMatrix globalBindposeInverseMatrix = mtxTransformLink.Inverse();
	#else
			// NOTE
			// https://github.com/cocos2d-x/fbx-conv/blob/master/src/readers/FbxConverter.h
			// getBindPose

			FbxAMatrix reference;
			cluster->GetTransformMatrix(reference);

			const FbxVector4 lT = node->GetGeometricTranslation(FbxNode::eSourcePivot);
			const FbxVector4 lR = node->GetGeometricRotation(FbxNode::eSourcePivot);
			const FbxVector4 lS = node->GetGeometricScaling(FbxNode::eSourcePivot);

			FbxAMatrix refGeom(lT, lR, lS);

			reference *= refGeom;

			FbxAMatrix init;
			cluster->GetTransformLinkMatrix(init);

			FbxAMatrix globalBindposeInverseMatrix = init.Inverse() * reference;
	#endif

			for (uint32_t i = 0; i < 4; i++) {
				for (uint32_t n = 0; n < 4; n++) {
					mtx.m[i][n] = static_cast<float>(globalBindposeInverseMatrix.Get(i, n));
				}
			}
		}
		else {
			mtx.identity();
		}
	}

	void FbxImporter::getJointTransform(
		uint32_t nIdx,
		const std::vector<aten::JointParam>& tvJoint,
		std::vector<JointTransformParam>& tvTransform)
	{
		FbxNode* node = m_dataMgr->getFbxNode(nIdx);

		auto name = node->GetName();

		FbxAMatrix& mtxLocal = node->EvaluateLocalTransform();

		const FbxVector4 trans = mtxLocal.GetT();
		const FbxQuaternion quat = mtxLocal.GetQ();
		const FbxVector4 scale = mtxLocal.GetS();

		// For quat.
		if (quat.mData[0] != 0.0f
			|| quat.mData[1] != 0.0f
			|| quat.mData[2] != 0.0f
			|| quat.mData[3] != 1.0f)
		{
			tvTransform.push_back(JointTransformParam());
			JointTransformParam& sTransform = tvTransform.back();

			sTransform.type = JointTransform::Quaternion;

			sTransform.param.push_back(static_cast<float>(quat.mData[0]));
			sTransform.param.push_back(static_cast<float>(quat.mData[1]));
			sTransform.param.push_back(static_cast<float>(quat.mData[2]));
			sTransform.param.push_back(static_cast<float>(quat.mData[3]));
		}

		// For trans.
		if (trans.mData[0] != 0.0f
			|| trans.mData[1] != 0.0f
			|| trans.mData[2] != 0.0f)
		{
			tvTransform.push_back(JointTransformParam());
			JointTransformParam& sTransform = tvTransform.back();

			sTransform.type = JointTransform::Translate;

			sTransform.param.push_back(static_cast<float>(trans.mData[0]));
			sTransform.param.push_back(static_cast<float>(trans.mData[1]));
			sTransform.param.push_back(static_cast<float>(trans.mData[2]));
		}

		// TODO
		// scale
	}

	//////////////////////////////////
	// For animation.

	bool FbxImporter::readBaseModel(const char* pszName)
	{
	#if 1
		m_dataMgrBase = new FbxDataManager();
		bool ret = m_dataMgrBase->openForAnm(pszName, true);

		return ret;
	#else
		return true;
	#endif
	}

	uint32_t FbxImporter::getAnmSetNum()
	{
		// NOTE
		// １しか許さない.
		return 1;
	}

	bool FbxImporter::beginAnm(uint32_t nSetIdx)
	{
		uint32_t m_curAnmIdx = nSetIdx;
		return true;
	}

	bool FbxImporter::endAnm()
	{
		return true;
	}

	uint32_t FbxImporter::getAnmNodeNum()
	{
		uint32_t ret = m_dataMgr->getNodeNum();

		if (m_dataMgrBase) {
			ret = m_dataMgr->reArrangeNodeByTargetBaseModel(m_dataMgrBase);
		}

		return ret;
	}

	uint32_t FbxImporter::getAnmChannelNum(uint32_t nNodeIdx)
	{
		static const fbxsdk::FbxVector4 vOne(1.0f, 1.0f, 1.0f, 0.0f);
		static const fbxsdk::FbxQuaternion qZero(0.0f, 0.0f, 0.0f, 1.0f);

		uint32_t num = 0;

		auto node = m_dataMgr->getFbxNode(nNodeIdx);

		auto start = m_dataMgr->getAnmStartFrame();
		auto stop = m_dataMgr->getAnmStopFrame();

		fbxsdk::FbxAMatrix prevMtx;
		prevMtx.SetIdentity();

		AnmChannel channel;
		channel.nodeIdx = nNodeIdx;

		for (int32_t f = start; f < stop; f++) {
			FbxTime time;
			time.Set(FbxTime::GetOneFrameValue(FbxTime::eFrames60) * f);

			auto mtx = node->EvaluateLocalTransform(time);

			if (mtx != prevMtx) {
				auto t = mtx.GetT();    // translate.
				auto s = mtx.GetS();    // scale.
				auto q = mtx.GetQ();    // quaternion.
				auto r = mtx.GetR();    // rotation.

				uint32_t pos = 0;

				// NOTE
				// Rotation -> Scale -> Translate

				// Rotate.
				if (!(q == qZero)) {
					num++;
					channel.type[pos++] = ParamType::Rotate;
				}
            
				// Scale.
				if (!(s == vOne)) {
					num++;
					channel.type[pos++] = ParamType::Scale;
				}

				// Trans.
				if (!t.IsZero(3)) {
					num++;
					channel.type[pos++] = ParamType::Tranlate;
				}

				break;
			}

			prevMtx = mtx;
		}

		m_channels.push_back(channel);

		return num;
	}

	bool FbxImporter::getAnmNode(
		uint32_t nNodeIdx,
		aten::AnmNode& sNode)
	{
		auto& node = m_dataMgr->getNode(nNodeIdx);
		auto fbxNode = node.fbxNode;

		AT_ASSERT(node.targetIdx >= 0);

		sNode.targetIdx = (node.targetIdx >= 0 ? node.targetIdx : nNodeIdx);

		sprintf(sNode.target, "%s\0", fbxNode->GetName());
		sNode.targetKey = aten::KeyValue::gen(sNode.target);

		sNode.numChannels = getAnmChannelNum(nNodeIdx);

		// NOTE
		// channelIdx は外部で設定される.

		return true;
	}

	bool FbxImporter::getAnmChannel(
		uint32_t nNodeIdx,
		uint32_t nChannelIdx,
		aten::AnmChannel& sChannel)
	{
		AT_ASSERT(nNodeIdx < m_channels.size());

		auto node = m_dataMgr->getFbxNode(nNodeIdx);

		auto& channel = m_channels[nNodeIdx];
		AT_ASSERT(channel.nodeIdx == nNodeIdx);

		// TODO
		// Not used.
		//sChannel.stride

		// NOTE
		// keyIdx は外部で設定される.

		if (!channel.isChecked)
		{
			auto start = m_dataMgr->getAnmStartFrame();
			auto stop = m_dataMgr->getAnmStopFrame();

			fbxsdk::FbxVector4 prevT;
			fbxsdk::FbxVector4 prevS(1.0f, 1.0f, 1.0f, 0.0f);
			fbxsdk::FbxQuaternion prevQ;

			for (int32_t f = start; f < stop; f++) {
				FbxTime time;
				time.Set(FbxTime::GetOneFrameValue(FbxTime::eFrames60) * f);

				auto mtx = node->EvaluateLocalTransform(time);

				auto t = mtx.GetT();    // translate.
				auto s = mtx.GetS();    // scale.
				auto q = mtx.GetQ();    // quaternion.
				auto r = mtx.GetR();    // rotation.

				// NOTE
				// Rotation -> Scale -> Translate

				// Rotate.
				if (!(q == prevQ)) {
					AnmKey key;
					key.key = f - start;

					key.value[0] = (float)q.mData[0];
					key.value[1] = (float)q.mData[1];
					key.value[2] = (float)q.mData[2];
					key.value[3] = (float)q.mData[3];

					channel.keys[ParamType::Rotate].push_back(key);
				}
				prevQ = q;

				// Scale.
				if (!(s == prevS)) {
					AnmKey key;
					key.key = f - start;

					key.value[0] = (float)s.mData[0];
					key.value[1] = (float)s.mData[1];
					key.value[2] = (float)s.mData[2];
					key.value[3] = (float)s.mData[3];

					channel.keys[ParamType::Scale].push_back(key);
				}
				prevS = s;

				// Trans.
				if (!(t == prevT)) {
					AnmKey key;
					key.key = f - start;

					key.value[0] = (float)t.mData[0];
					key.value[1] = (float)t.mData[1];
					key.value[2] = (float)t.mData[2];
					key.value[3] = (float)t.mData[3];

					channel.keys[ParamType::Tranlate].push_back(key);
				}
				prevT = t;
			}

			// NOTE
			// キーデータが１つだと補間できないので、あえて増やす.
			for (uint32_t i = 0; i < ParamType::Num; i++) {
				if (channel.keys[i].size() == 1) {
					auto key = channel.keys[i][0];
					key.key = (stop - start) - 1;
					channel.keys[i].push_back(key);
				}
			}
		}

		channel.isChecked = true;

		auto type = channel.type[nChannelIdx];
    
		// NOTE
		// Rotation -> Scale -> Translate

		// Rotation
		if (type == ParamType::Rotate) {
			sChannel.numKeys = (uint16_t)channel.keys[type].size();
			sChannel.interp = aten::AnmInterpType::Slerp;
			sChannel.type = aten::AnmTransformType::QuaternionXYZW;
			return true;
		}

		// Scale
		if (type == ParamType::Scale) {
			sChannel.numKeys = (uint16_t)channel.keys[type].size();
			sChannel.interp = aten::AnmInterpType::Linear;
			sChannel.type = aten::AnmTransformType::ScaleXYZ;
			return true;
		}

		// Translate
		if (type == ParamType::Tranlate) {
			sChannel.numKeys = (uint16_t)channel.keys[type].size();
			sChannel.interp = aten::AnmInterpType::Linear;
			sChannel.type = aten::AnmTransformType::TranslateXYZ;
			return true;
		}

		return false;
	}

	bool FbxImporter::getAnmKey(
		uint32_t nNodeIdx,
		uint32_t nChannelIdx,
		uint32_t nKeyIdx,
		aten::AnmKey& sKey,
		std::vector<float>& tvValue)
	{
		AT_ASSERT(nNodeIdx < m_channels.size());

		auto node = m_dataMgr->getFbxNode(nNodeIdx);

		auto& channel = m_channels[nNodeIdx];
		AT_ASSERT(channel.nodeIdx == nNodeIdx);

		auto type = channel.type[nChannelIdx];

		// TODO
		// 60FPS固定.
		static const float frame = 1.0f / 60.0f;

		AT_ASSERT(nKeyIdx < channel.keys[type].size());
		const auto& key = channel.keys[type][nKeyIdx];

		if (nKeyIdx > 0) {
			AT_ASSERT(key.key > channel.keys[type][nKeyIdx - 1].key);
		}

		sKey.keyTime = key.key * frame;

		// NOTE
		// Rotation -> Scale -> Translate

		// Rotation
		if (type == ParamType::Rotate) {
			sKey.numParams = 4;

			tvValue.push_back(key.value[0]);
			tvValue.push_back(key.value[1]);
			tvValue.push_back(key.value[2]);
			tvValue.push_back(key.value[3]);

			return true;
		}

		// Scale
		if (type == ParamType::Scale) {
			sKey.numParams = 3;

			tvValue.push_back(key.value[0]);
			tvValue.push_back(key.value[1]);
			tvValue.push_back(key.value[2]);

			return true;
		}

		// Translate
		if (type == ParamType::Tranlate) {
			sKey.numParams = 3;

			tvValue.push_back(key.value[0]);
			tvValue.push_back(key.value[1]);
			tvValue.push_back(key.value[2]);

			return true;
		}

		return false;
	}

	//////////////////////////////////
	// For material.

	bool FbxImporter::beginMaterial()
	{
		return true;
	}

	bool FbxImporter::endMaterial()
	{
		return true;
	}

	uint32_t FbxImporter::getMaterialNum()
	{
		m_dataMgr->loadMaterial();
		uint32_t ret = m_dataMgr->getMaterialNum();
		return ret;
	}

	static const char* FbxMtrlParamNames[] = {
		fbxsdk::FbxSurfaceMaterial::sEmissive,
		fbxsdk::FbxSurfaceMaterial::sEmissiveFactor,

		fbxsdk::FbxSurfaceMaterial::sAmbient,
		fbxsdk::FbxSurfaceMaterial::sAmbientFactor,

		fbxsdk::FbxSurfaceMaterial::sDiffuse,
		fbxsdk::FbxSurfaceMaterial::sDiffuseFactor,

		fbxsdk::FbxSurfaceMaterial::sSpecular,
		fbxsdk::FbxSurfaceMaterial::sSpecularFactor,
		fbxsdk::FbxSurfaceMaterial::sShininess,

		fbxsdk::FbxSurfaceMaterial::sBump,
		fbxsdk::FbxSurfaceMaterial::sNormalMap,
		fbxsdk::FbxSurfaceMaterial::sBumpFactor,

		fbxsdk::FbxSurfaceMaterial::sTransparentColor,
		fbxsdk::FbxSurfaceMaterial::sTransparencyFactor,

		fbxsdk::FbxSurfaceMaterial::sReflection,
		fbxsdk::FbxSurfaceMaterial::sReflectionFactor,

		fbxsdk::FbxSurfaceMaterial::sDisplacementColor,
		fbxsdk::FbxSurfaceMaterial::sDisplacementFactor,

		fbxsdk::FbxSurfaceMaterial::sVectorDisplacementColor,
		fbxsdk::FbxSurfaceMaterial::sVectorDisplacementFactor,
	};

	void FbxImporter::getLambertParams(
		void* mtrl, 
		std::vector<MaterialParam>& list)
	{
		fbxsdk::FbxSurfaceLambert* lambert = (fbxsdk::FbxSurfaceLambert*)mtrl;

		if (lambert->Diffuse.IsValid()) {
			MaterialParam diffuse;
			diffuse.name = "diffuse";
			diffuse.values.push_back((float)lambert->Diffuse.Get().mData[0]);
			diffuse.values.push_back((float)lambert->Diffuse.Get().mData[1]);
			diffuse.values.push_back((float)lambert->Diffuse.Get().mData[2]);
			diffuse.values.push_back(1.0f);
			list.push_back(diffuse);
		}

		if (lambert->Ambient.IsValid()) {
			MaterialParam ambient;
			ambient.name = "ambient";
			ambient.values.push_back((float)lambert->Ambient.Get().mData[0]);
			ambient.values.push_back((float)lambert->Ambient.Get().mData[1]);
			ambient.values.push_back((float)lambert->Ambient.Get().mData[2]);
			ambient.values.push_back(1.0f);
			list.push_back(ambient);
		}

		if (lambert->Emissive.IsValid()) {
			MaterialParam emmisive;
			emmisive.name = "emmisive";
			emmisive.values.push_back((float)lambert->Emissive.Get().mData[0]);
			emmisive.values.push_back((float)lambert->Emissive.Get().mData[1]);
			emmisive.values.push_back((float)lambert->Emissive.Get().mData[2]);
			emmisive.values.push_back(1.0f);
			list.push_back(emmisive);
		}
	}

	void FbxImporter::getPhongParams(
		void* mtrl, 
		std::vector<MaterialParam>& list)
	{
		getLambertParams(mtrl, list);

		fbxsdk::FbxSurfacePhong* phong = (fbxsdk::FbxSurfacePhong*)mtrl;

		if (phong->Specular.IsValid()) {
			MaterialParam specular;
			specular.name = "specular";
			specular.values.push_back((float)phong->Specular.Get().mData[0]);
			specular.values.push_back((float)phong->Specular.Get().mData[1]);
			specular.values.push_back((float)phong->Specular.Get().mData[2]);
			specular.values.push_back(1.0f);
			list.push_back(specular);
		}

		if (phong->Shininess.IsValid()) {
			MaterialParam shiness;
			shiness.name = "shiness";
			shiness.values.push_back((float)phong->Shininess.Get());
			list.push_back(shiness);
		}
	}

	bool FbxImporter::getMaterial(
		uint32_t nMtrlIdx,
		MaterialInfo& mtrl)
	{
		auto* fbxMtrl = m_dataMgr->getMaterial(nMtrlIdx);

		mtrl.name = fbxMtrl->GetName();

		bool ret = false;

		// TODO
		// cgfx.
		auto implementation = GetImplementation(fbxMtrl, FBXSDK_IMPLEMENTATION_CGFX);

		if (implementation != nullptr) {
			ret = getFbxMatrialByImplmentation(nMtrlIdx, mtrl.tex, mtrl.params);
		}
		else {
			ret = getFbxMatrial(nMtrlIdx, mtrl.tex, mtrl.params);
		}

		return ret;
	}

	bool FbxImporter::getFbxMatrial(
		uint32_t nMtrlIdx,
		std::vector<MaterialTex>& mtrlTex,
		std::vector<MaterialParam>& mtrlParam)
	{
		auto* fbxMtrl = m_dataMgr->getMaterial(nMtrlIdx);

		std::vector<MaterialTex> texList;

		// for tex.
		for (uint32_t i = 0; i < AT_COUNTOF(FbxMtrlParamNames); i++)
		{
			std::string name(FbxMtrlParamNames[i]);

			fbxsdk::FbxProperty prop = fbxMtrl->FindProperty(name.c_str());
			if (!prop.IsValid()) {
				continue;
			}

			// NOTE
			// http://stackoverflow.com/questions/19634369/read-texture-filename-from-fbx-with-fbx-sdk-c
			// http://marupeke296.com/FBX_No7_TextureMaterial.html

			// プロパティが持っているレイヤードテクスチャの枚数をチェック.
			int layerNum = prop.GetSrcObjectCount<fbxsdk::FbxLayeredTexture>();

			if (layerNum > 0) {
				// TODO
			}
			else {
				int textureCount = prop.GetSrcObjectCount<fbxsdk::FbxTexture>();

				for (int n = 0; n < textureCount; n++)
				{
					fbxsdk::FbxTexture* texture = FbxCast<fbxsdk::FbxTexture>(prop.GetSrcObject<fbxsdk::FbxTexture>(n));

					MaterialTex tex;
					{
						tex.name = ((fbxsdk::FbxFileTexture*)texture)->GetRelativeFileName();
					}

					if (name == fbxsdk::FbxSurfaceMaterial::sSpecular) {
						tex.type.isSpecular = true;
					}
					else if (name == fbxsdk::FbxSurfaceMaterial::sNormalMap) {
						tex.type.isNormal = true;
					}
					else if (name == fbxsdk::FbxSurfaceMaterial::sTransparentColor) {
						tex.type.isTranslucent = true;
					}

					// TODO
					// 他の場合...

					mtrlTex.push_back(tex);
				}
			}
		}

		// for param.
		if (fbxMtrl->GetClassId().Is(fbxsdk::FbxSurfacePhong::ClassId)) {
			getPhongParams(fbxMtrl, mtrlParam);
		}
		else if (fbxMtrl->GetClassId().Is(fbxsdk::FbxSurfaceLambert::ClassId)) {
			getLambertParams(fbxMtrl, mtrlParam);
		}
		else {
			AT_ASSERT(false);
		}

		return true;
	}

	bool FbxImporter::getFbxMatrialByImplmentation(
		uint32_t nMtrlIdx,
		std::vector<MaterialTex>& mtrlTex,
		std::vector<MaterialParam>& mtrlParam)
	{
		// NOTE
		// http://www.programmershare.com/3142984/

		auto* fbxMtrl = m_dataMgr->getMaterial(nMtrlIdx);

		// TODO
		// cgfx.
		auto implementation = GetImplementation(fbxMtrl, FBXSDK_IMPLEMENTATION_CGFX);
		AT_VRETURN_FALSE(implementation != nullptr);

		auto rootTable = implementation->GetRootTable();
		auto entryCount = rootTable->GetEntryCount();

		std::vector<MaterialTex> tmpTexList;

		for (int i = 0; i < entryCount; ++i)
		{
			auto entry = rootTable->GetEntry(i);

			auto fbxProperty = fbxMtrl->FindPropertyHierarchical(entry.GetSource());
			if (!fbxProperty.IsValid()) {
				fbxProperty = fbxMtrl->RootProperty.FindHierarchical(entry.GetSource());
			}

			auto propName = fbxProperty.GetNameAsCStr();

			auto textureCount = fbxProperty.GetSrcObjectCount<FbxTexture>();

			if (textureCount > 0)
			{
				// for tex.

				std::string src = entry.GetSource();

				auto num = fbxProperty.GetSrcObjectCount<FbxFileTexture>();

				for (int n = 0; n < num; n++)
				{
					auto texFile = fbxProperty.GetSrcObject<FbxFileTexture>(n);
					std::string texName = texFile->GetFileName();
					texName = texName.substr(texName.find_last_of('/') + 1);

					auto texture = fbxProperty.GetSrcObject<FbxTexture>(n);

					MaterialTex tex;
					{
						tex.name = texName;
					}

					if (src == "Maya|DiffuseTexture") {
						// Nothing.
					}
					else if (src == "Maya|NormalTexture") {
						tex.type.isNormal = true;
					}
					else if (src == "Maya|SpecularTexture") {
						tex.type.isSpecular = true;
					}
					else if (src == "Maya|FalloffTexture") {
						// TODO
					}
					else if (src == "Maya|ReflectionMapTexture") {
						// TODO
					}

					tmpTexList.push_back(tex);
				}
			}
			else {
				// for param.

				auto dataType = fbxProperty.GetPropertyDataType().GetType();

				MaterialParam param;
				param.name = propName;

				if (dataType == fbxsdk::eFbxBool) {
					bool v = fbxProperty.Get<bool>();
					param.values.push_back(v);
				}
				else if (dataType == fbxsdk::eFbxInt || dataType == fbxsdk::eFbxEnum) {
					int v = fbxProperty.Get<int>();
					param.values.push_back((float)v);
				}
				else if (dataType == fbxsdk::eFbxUInt) {
					unsigned int v = fbxProperty.Get<unsigned int>();
					param.values.push_back((float)v);
				}
				else if (dataType == fbxsdk::eFbxFloat) {
					float v = fbxProperty.Get<float>();
					param.values.push_back(v);
				}
				else if (dataType == fbxsdk::eFbxDouble) {
					double v = fbxProperty.Get<double>();
					param.values.push_back((float)v);
				}
				else if (dataType == fbxsdk::eFbxDouble2) {
					FbxDouble2 v = fbxProperty.Get<FbxDouble2>();
					param.values.push_back((float)v.mData[0]);
					param.values.push_back((float)v.mData[1]);
				}
				else if (dataType == fbxsdk::eFbxDouble3) {
					FbxDouble3 v = fbxProperty.Get<FbxDouble3>();
					param.values.push_back((float)v.mData[0]);
					param.values.push_back((float)v.mData[1]);
					param.values.push_back((float)v.mData[2]);
				}
				else if (dataType == fbxsdk::eFbxDouble4) {
					FbxDouble4 v = fbxProperty.Get<FbxDouble4>();
					param.values.push_back((float)v.mData[0]);
					param.values.push_back((float)v.mData[1]);
					param.values.push_back((float)v.mData[2]);
					param.values.push_back((float)v.mData[3]);
				}
				else if (dataType == fbxsdk::eFbxDouble4x4) {
					FbxDouble4x4 v = fbxProperty.Get<FbxDouble4x4>();

					for (int i = 0; i < 4; i++) {
						for (int n = 0; n < 4; n++) {
							param.values.push_back((float)v.mData[i].mData[n]);
						}
					}
				}
				else {
					AT_ASSERT(false);
				}

				if (param.values.size() > 0) {
					mtrlParam.push_back(param);
				}
			}
		}

		for (uint32_t i = 0; i < (uint32_t)tmpTexList.size(); i++) {
			if (m_ignoreTexIdx == i) {
				continue;
			}

			mtrlTex.push_back(tmpTexList[i]);
		}

		return true;
	}
}