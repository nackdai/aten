#include "aten.h"
#include "atenscene.h"
#include "ObjWriter.h"

#include <cmdline.h>

struct Options {
	std::string input;
	std::string output;
};

bool parseOption(
	int argc, char* argv[],
	cmdline::parser& cmd,
	Options& opt)
{
	{
		cmd.add<std::string>("input", 'i', "input filename", true);
		cmd.add<std::string>("output", 'o', "output filename base", false, "result");

		cmd.add<std::string>("help", '?', "print usage", false);
	}

	bool isCmdOk = cmd.parse(argc, argv);

	if (cmd.exist("help")) {
		std::cerr << cmd.usage();
		return false;
	}

	if (!isCmdOk) {
		std::cerr << cmd.error() << std::endl << cmd.usage();
		return false;
	}

	if (cmd.exist("input")) {
		opt.input = cmd.get<std::string>("input");
	}
	else {
		std::cerr << cmd.error() << std::endl << cmd.usage();
		return false;
	}

	if (cmd.exist("output")) {
		opt.output = cmd.get<std::string>("output");
	}
	else {
		// TODO
		opt.output = "result.sbvh";
	}

	return true;
}

int main(int argc, char* argv[])
{
	Options opt;
	cmdline::parser cmd;

#if 0
	if (!parseOption(argc, argv, cmd, opt)) {
		return 0;
	}
#endif

	aten::window::SetCurrentDirectoryFromExe();

	aten::AssetManager::suppressWarnings();
	aten::ObjLoader::disableGenGLTexture();

	std::vector<aten::object*> objs;
	aten::ObjLoader::load(
		objs,
		"../../asset/mansion/interior_bundled4_chairmove_1163769_606486_2.obj");

	if (objs.empty()) {
		// TODO
		return 0;
	}

	// TODO
	static const char* names[] = {
		"Screen",
		"base",
		"Keyboard",
		"Middle",
		"Silver",
	};

	auto obj = objs[0];

	std::vector<aten::objshape*> shapes;
	std::vector<aten::material*> mtrls;

	for (auto shape : obj->shapes) {
		auto mtrl = shape->getMaterial();

		std::string mtrlName(mtrl->name());

		for (auto name : names) {
			if (mtrlName == name) {
				shapes.push_back(shape);
				mtrls.push_back(const_cast<aten::material*>(mtrl));
				
				break;
			}
		}
	}

	struct IndexMapper {
		int orgIdx;
		int newIdx;

		IndexMapper(int o, int n) : orgIdx(o), newIdx(n) {}
	};

	std::vector<IndexMapper> indexMapList;

	// Gather vertices.
	std::vector<aten::vertex> vertices;
	{
		// Gather indices for gathering vertices.
		std::vector<int> tmpIndices(shapes.size());

		for (auto shape : shapes) {
			const auto& tris = shape->tris();

			for (auto tri : tris) {
				for (int i = 0; i < 3; i++) {
					auto idx = tri->param.idx[i];
					auto found = std::find(tmpIndices.begin(), tmpIndices.end(), idx);
					if (found == tmpIndices.end()) {
						tmpIndices.push_back(idx);
					}
				}
			}
		}

		indexMapList.reserve(tmpIndices.size());

		// Gather vertices.
		const auto& vtxs = aten::VertexManager::getVertices();
		for (auto idx : tmpIndices) {
			const auto v = vtxs[idx];
			
			int newIdx = vertices.size();
			vertices.push_back(v);

			// Keep index map.
			indexMapList.push_back(IndexMapper(idx, newIdx));
		}
	}

	// Gather indices.
	std::vector<std::vector<int>> indices(shapes.size());
	{
		for (int s = 0; s < shapes.size(); s++) {
			auto shape = shapes[s];
			const auto& tris = shape->tris();

			for (auto tri : tris) {
				for (int i = 0; i < 3; i++) {
					auto idx = tri->param.idx[i];

					auto found = std::find_if(
						indexMapList.begin(),
						indexMapList.end(),
						[&](const IndexMapper& idxMapper)
					{
						return idxMapper.orgIdx == idx;
					});

					AT_ASSERT(found != indexMapList.end());

					int newIdx = found->newIdx;
					indices[s].push_back(newIdx);
				}
			}
		}
	}

	// Export to obj.
	ObjWriter::write(
		"result.obj",
		"result.mtl",
		vertices,
		indices,
		mtrls);

	return 1;
}
