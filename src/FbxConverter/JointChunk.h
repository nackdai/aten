#pragma once

#include "types.h"
#include "FbxImporter.h"

#include "deformable/SKLFormat.h"

#include <vector>

class FileOutputStream;

class JointChunk {
protected:
	JointChunk();
    ~JointChunk();

public:
    static bool exportJoint(
        FileOutputStream* pOut,
        aten::FbxImporter* pImporter);

protected:
    static void getJointInfo(
        FileOutputStream* pOut,
		aten::FbxImporter* pImporter,
        std::vector<aten::JointParam>& tvJoint);

    static bool exportJoint(
        FileOutputStream* pOut,
		aten::FbxImporter* pImporter,
        std::vector<aten::JointParam>& tvJoint);
};
