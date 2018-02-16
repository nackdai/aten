#pragma once

#include "types.h"
#include "FbxImporter.h"
#include <vector>

class FileOutputStream;

class JointChunk {
protected:
	JointChunk();
    ~JointChunk();

public:
    static bool export(
        FileOutputStream* pOut,
        aten::FbxImporter* pImporter);

protected:
    static void getJointInfo(
        FileOutputStream* pOut,
		aten::FbxImporter* pImporter,
        std::vector<izanagi::S_SKL_JOINT>& tvJoint);

    static bool exportJoint(
        FileOutputStream* pOut,
		aten::FbxImporter* pImporter,
        std::vector<izanagi::S_SKL_JOINT>& tvJoint);
};
