#include "aten.h"
#include "atenscene.h"

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

	if (!parseOption(argc, argv, cmd, opt)) {
		return 0;
	}

	aten::window::SetCurrentDirectoryFromExe();

	aten::AssetManager::suppressWarnings();

	auto obj = aten::ObjLoader::load(opt.input);

	if (!obj) {
		// TODO
		return 0;
	}

	// Not use.
	// But specify internal accelerator type in constructor...
	aten::AcceleratedScene<aten::sbvh> scene;
	
	try {
		obj->exportInternalAccelTree(opt.output.c_str());
	}
	catch (std::exception* e) {
		// TODO

		return 0;
	}

	return 1;
}
