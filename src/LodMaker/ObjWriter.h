#pragma once

#include "aten.h"
#include <vector>

class ObjWriter {
public:
	ObjWriter() {}
	~ObjWriter() {}

public:
	static bool write(
		const char* path,
		const std::vector<aten::vertex>& vertices,
		const std::vector<std::vector<int>>& indices,
		const std::vector<aten::material*>& mtrls);

	bool runOnThread(
		std::function<void()> funcFinish,
		const char* path,
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

		const char* path;
		const std::vector<aten::vertex>& vertices;
		const std::vector<std::vector<int>>& indices;
		const std::vector<aten::material*>& mtrls;

		WriteParams(
			std::function<void()> _func,
			const char* _path,
			const std::vector<aten::vertex>& _vertices,
			const std::vector<std::vector<int>>& _indices,
			const std::vector<aten::material*>& _mtrls)
			: vertices(_vertices), indices(_indices), mtrls(_mtrls)
		{
			funcFinish = _func;
			path = _path;
		}
	};

	WriteParams* m_param{ nullptr };

	aten::Thread m_thread;
	aten::Semaphore m_sema;

	std::atomic<bool> m_isRunning{ false };
	std::atomic<bool> m_isTerminate{ false };
};
