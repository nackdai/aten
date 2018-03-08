#pragma once

#include "aten.h"
#include <vector>

class ObjWriter {
public:
	ObjWriter() {}
	~ObjWriter() {}

public:
	static bool write(
		const std::string& path,
		const std::string& mtrlName,
		const std::vector<aten::vertex>& vertices,
		const std::vector<std::vector<int>>& indices,
		const std::vector<aten::material*>& mtrls);

	bool runOnThread(
		std::function<void()> funcFinish,
		const std::string& path,
		const std::string& mtrlName,
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
		std::string mtrlName;
		const std::vector<aten::vertex>& vertices;
		const std::vector<std::vector<int>>& indices;
		const std::vector<aten::material*>& mtrls;

		WriteParams(
			std::function<void()> _func,
			const std::string& _path,
			const std::string& _mtrlName,
			const std::vector<aten::vertex>& _vertices,
			const std::vector<std::vector<int>>& _indices,
			const std::vector<aten::material*>& _mtrls)
			: path(_path), vertices(_vertices), indices(_indices), mtrls(_mtrls)
		{
			funcFinish = _func;
			path = _path;
			mtrlName = _mtrlName;
		}
	};

	WriteParams* m_param{ nullptr };

	aten::Thread m_thread;
	aten::Semaphore m_sema;

	std::atomic<bool> m_isRunning{ false };
	std::atomic<bool> m_isTerminate{ false };
};
