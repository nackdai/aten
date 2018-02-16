
#ifndef NV_TRISTRIP_OBJECTS_H
#define NV_TRISTRIP_OBJECTS_H

#include <assert.h>
#include <windows.h>
#include <vector>
#include <list>
#include "VertexCache.h"

/////////////////////////////////////////////////////////////////////////////////
//
// Types defined for stripification
//
/////////////////////////////////////////////////////////////////////////////////

struct MyVertex {
	float x, y, z;
	float nx, ny, nz;
};

typedef MyVertex MyVector;

struct MyFace {
	int v1, v2, v3;
	float nx, ny, nz;
};

/////////////////////////////////////////////////////////////////////////////

class NvFaceInfo {
public:
	
	// vertex indices
	NvFaceInfo(int v0, int v1, int v2, bool bIsFake = false){
		m_v0 = v0; m_v1 = v1; m_v2 = v2;
		m_stripId      = -1;
		m_testStripId  = -1;
		m_experimentId = -1;
		m_bIsFake = bIsFake;
	}

	bool IsContain(int v0, int v1, int v2);
	
	// data members are left public
public:
	int   m_v0, m_v1, m_v2;

	// real strip Id
	// 所属するストリップID
	int   m_stripId;

	// strip Id in an experiment
	// お試し中のときに所属するストリップID
	int   m_testStripId;

	// in what experiment was it given an experiment Id?
	// お試し中ID
	int   m_experimentId;

	//if true, will be deleted when the strip it's in is deleted
	// 縮退三角形かどうか
	bool  m_bIsFake;
};

inline bool NvFaceInfo::IsContain(int v0, int v1, int v2)
{
	bool ret = (((m_v0 == v0) || (m_v0 == v1) || (m_v0 == v2))
				&& ((m_v1 == v0) || (m_v1 == v1) || (m_v1 == v2))
				&& ((m_v2 == v0) || (m_v2 == v1) || (m_v2 == v2)));
	return ret;
}

/////////////////////////////////////////////////////////////////////////////
// nice and dumb edge class that points knows its
// indices, the two faces, and the next edge using
// the lesser of the indices
class NvEdgeInfo {
public:
	
	// constructor puts 1 ref on us
	NvEdgeInfo (int v0, int v1){
		m_v0       = v0;
		m_v1       = v1;
		m_face0    = NULL;
		m_face1    = NULL;
		m_nextV0   = NULL;
		m_nextV1   = NULL;
		
		// we will appear in 2 lists.  this is a good
		// way to make sure we delete it the second time
		// we hit it in the edge infos
		m_refCount = 2;    
		
	}
	
	// ref and unref
	void Unref () { if (--m_refCount == 0) delete this; }
	
	// data members are left public
	UINT         m_refCount;			// 参照カウンタ
	NvFaceInfo  *m_face0, *m_face1;		// この辺を持つ面
	int          m_v0, m_v1;			// 辺を構成する頂点ID
	NvEdgeInfo  *m_nextV0, *m_nextV1;	// 隣接する辺
};

/////////////////////////////////////////////////////////////////////////////
// This class is a quick summary of parameters used to begin a triangle strip.
// Some operations may want to create lists of such items,
// so they were pulled out into a class
class NvStripStartInfo {
public:
	NvStripStartInfo(NvFaceInfo *startFace, NvEdgeInfo *startEdge, bool toV1){
		m_startFace    = startFace;
		m_startEdge    = startEdge;
		m_toV1         = toV1;
	}
	NvFaceInfo    *m_startFace;	// ストリップ開始面
	NvEdgeInfo    *m_startEdge;	// ストリップ開始辺
	bool           m_toV1;      // 辺を構成する頂点の順番を正確に取得するためのフラグ(true -> 01 / false -> 10)
};

/////////////////////////////////////////////////////////////////////////////

typedef std::vector<NvFaceInfo*>     NvFaceInfoVec;
typedef std::list  <NvFaceInfo*>     NvFaceInfoList;
typedef std::list  <NvFaceInfoVec*>  NvStripList;
typedef std::vector<NvEdgeInfo*>     NvEdgeInfoVec;

#if 0
typedef std::vector<DWORD> WordVec;
#else
typedef std::vector<unsigned int> WordVec;
#endif
typedef std::vector<int> IntVec;
typedef std::vector<MyVertex> MyVertexVec;
typedef std::vector<MyFace> MyFaceVec;

template<class T> 
inline void SWAP(T& first, T& second) 
{
	T temp = first;
	first = second;
	second = temp;
}

/////////////////////////////////////////////////////////////////////////////
// This is a summary of a strip that has been built
class NvStripInfo {
public:
	
	// A little information about the creation of the triangle strips
	NvStripInfo(const NvStripStartInfo &startInfo, int stripId, int experimentId = -1)
		: m_startInfo(startInfo)
	{
		m_stripId      = stripId;
		m_experimentId = experimentId;
		visited = false;
		m_numDegenerates = 0;
	}

	// This is an experiment if the experiment id is >= 0
	inline bool IsExperiment () const;
	  
	inline bool IsInStrip (const NvFaceInfo *faceInfo) const;
	  
	bool SharesEdge(const NvFaceInfo* faceInfo, NvEdgeInfoVec &edgeInfos);
	  
	// take the given forward and backward strips and combine them together
	void Combine(const NvFaceInfoVec &forward, const NvFaceInfoVec &backward);
	  
	//returns true if the face is "izique", i.e. has a vertex which doesn't exist in the faceVec
	bool Unique(NvFaceInfoVec& faceVec, NvFaceInfo* face);
	  
	// mark the triangle as taken by this strip
	bool IsMarked    (NvFaceInfo *faceInfo);
	void MarkTriangle(NvFaceInfo *faceInfo);
	  
	// build the strip
	void Build(NvEdgeInfoVec &edgeInfos, NvFaceInfoVec &faceInfos);
	  
	// public data members
	NvStripStartInfo m_startInfo;
	NvFaceInfoVec    m_faces;			// ストリップに含まれる面リスト
	int              m_stripId;			// ストリップID
	int              m_experimentId;	// お試し中ID
	  
	bool visited;

	int m_numDegenerates;				// 縮退三角形数
};

// This is an experiment if the experiment id is >= 0
// お試し中かどうか
bool NvStripInfo::IsExperiment () const
{
	return m_experimentId >= 0;
}

// ストリップに含まれるかどうか
bool NvStripInfo::IsInStrip (const NvFaceInfo *faceInfo) const
{
	if(faceInfo == NULL)
		return false;
	  
	return (m_experimentId >= 0 ? faceInfo->m_testStripId == m_stripId : faceInfo->m_stripId == m_stripId);
}

typedef std::vector<NvStripInfo*>    NvStripInfoVec;

/////////////////////////////////////////////////////////////////////////////
//The actual stripifier
class NvStripifier {
public:
	
	// Constructor
	NvStripifier();
	~NvStripifier();
	
	//the target vertex cache size, the structure to place the strips in, and the input indices
	void Stripify(
		const WordVec &in_indices,
		const int in_cacheSize,
		const int in_minStripLength, 
		const unsigned int maxIndex,
		NvStripInfoVec &allStrips,
		NvFaceInfoVec &allFaces);

	void CreateStrips(
		const NvStripInfoVec& allStrips,
		IntVec& stripIndices,
		const bool bStitchStrips,
		unsigned int& numSeparateStrips,
		const bool bRestart,
		const unsigned int restartVal);
	
	static int GetUniqueVertexInB(NvFaceInfo *faceA, NvFaceInfo *faceB);
	//static int GetSharedVertex(NvFaceInfo *faceA, NvFaceInfo *faceB);
	static void GetSharedVertices(
		NvFaceInfo *faceA, NvFaceInfo *faceB,
		int* vertex0, int* vertex1);

	static bool IsDegenerate(const NvFaceInfo* face);
	static bool IsDegenerate(const unsigned int v0, const unsigned int v1, const unsigned int v2);
	
protected:
	
	WordVec m_Indices;
	int m_nCacheSize;
	int m_nMinStripLength;
	float m_fMeshJump;
	bool m_bFirstTimeResetPoint;
	
	/////////////////////////////////////////////////////////////////////////////////
	//
	// Big mess of functions called during stripification
	//
	/////////////////////////////////////////////////////////////////////////////////

	//********************
	bool IsMoneyFace(const NvFaceInfo& face);
	bool FaceContainsIndex(const NvFaceInfo& face, const unsigned int index);

	bool IsCW(NvFaceInfo *faceInfo, int v0, int v1);
	bool NextIsCW(const int numIndices);
	
	static int  GetNextIndex(const WordVec &indices, NvFaceInfo *face);
	static NvEdgeInfo *FindEdgeInfo(NvEdgeInfoVec &edgeInfos, int v0, int v1);
	static NvFaceInfo *FindOtherFace(NvEdgeInfoVec &edgeInfos, int v0, int v1, NvFaceInfo *faceInfo);
	NvFaceInfo *FindGoodResetPoint(NvFaceInfoVec &faceInfos, NvEdgeInfoVec &edgeInfos);
	
	void FindAllStrips(NvStripInfoVec &allStrips, NvFaceInfoVec &allFaceInfos, NvEdgeInfoVec &allEdgeInfos, int numSamples);
	void SplitUpStripsAndOptimize(NvStripInfoVec &allStrips, NvStripInfoVec &outStrips, NvEdgeInfoVec& edgeInfos, NvFaceInfoVec& outFaceList);
	void RemoveSmallStrips(NvStripInfoVec& allStrips, NvStripInfoVec& allBigStrips, NvFaceInfoVec& faceList);
	
	bool FindTraversal(NvFaceInfoVec &faceInfos, NvEdgeInfoVec &edgeInfos, NvStripInfo *strip, NvStripStartInfo &startInfo);
	int  CountRemainingTris(std::list<NvStripInfo*>::iterator iter, std::list<NvStripInfo*>::iterator  end);
	
	void CommitStrips(NvStripInfoVec &allStrips, const NvStripInfoVec &strips);
	
	float AvgStripSize(const NvStripInfoVec &strips);
	int FindStartPoint(NvFaceInfoVec &faceInfos, NvEdgeInfoVec &edgeInfos);
	
	void UpdateCacheStrip(VertexCache* vcache, NvStripInfo* strip);
	void UpdateCacheFace(VertexCache* vcache, NvFaceInfo* face);
	float CalcNumHitsStrip(VertexCache* vcache, NvStripInfo* strip);
	int CalcNumHitsFace(VertexCache* vcache, NvFaceInfo* face);
	int NumNeighbors(NvFaceInfo* face, NvEdgeInfoVec& edgeInfoVec);
	
	void BuildStripifyInfo(NvFaceInfoVec &faceInfos, NvEdgeInfoVec &edgeInfos, const unsigned int maxIndex);
	bool AlreadyExists(NvFaceInfo* faceInfo, NvFaceInfoVec& faceInfos);
	
	// let our strip info classes and the other classes get
	// to these protected stripificaton methods if they want
	friend NvStripInfo;
};

#endif
