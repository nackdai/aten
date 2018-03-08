#pragma once

#include "aten.h"
#include <vector>

// TODO
// namespace ‚ÅˆÍ‚¤‚Æ fatal error c1001 ‚ª”­¶‚·‚é.
// Œ´ˆö‚ª‚í‚©‚ç‚È‚¢‚½‚ßAwork around ‚Å‚Í‚ ‚é‚ª namespace ‚ğg—p‚µ‚È‚¢‚æ‚¤‚É‚·‚é...

class ObjWriter {
public:
	ObjWriter() {}
	~ObjWriter() {}

public:
	static bool write(
		const std::string& path,
		const std::string& mtrlPath,
		const std::vector<aten::vertex>& vertices,
		const std::vector<std::vector<int>>& indices,
		const std::vector<aten::material*>& mtrls);

	static bool writeMaterial(
		const std::string& mtrlPath,
		const std::vector<aten::material*>& mtrls);

	bool runOnThread(
		std::function<void()> funcFinish,
		const std::string& path,
		const std::string& mtrlPath,
		const std::vector<aten::vertex>& vertices,
		const std::vector<std::vector<int>>& indices,
		const std::vector<aten::material*>& mtrls);

	void terminate();

	bool isRunningThread() const
	{
		return m_isRunning;
	}

private:
	struct WriteParams {
		std::function<void()> funcFinish;

		std::string path;
		std::string mtrlPath;
		const std::vector<aten::vertex>& vertices;
		const std::vector<std::vector<int>>& indices;
		const std::vector<aten::material*>& mtrls;

		WriteParams(
			std::function<void()> _func,
			const std::string& _path,
			const std::string& _mtrlPath,
			const std::vector<aten::vertex>& _vertices,
			const std::vector<std::vector<int>>& _indices,
			const std::vector<aten::material*>& _mtrls)
			: path(_path), vertices(_vertices), indices(_indices), mtrls(_mtrls)
		{
			funcFinish = _func;
			path = _path;
			mtrlPath = _mtrlPath;
		}
	};

	WriteParams* m_param{ nullptr };

	aten::Thread m_thread;
	aten::Semaphore m_sema;

	std::atomic<bool> m_isRunning{ false };
	std::atomic<bool> m_isTerminate{ false };
};
