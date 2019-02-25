#include <cmdline.h>

#include "aten.h"

struct Options {
};

bool parseOption(
    int argc, char* argv[],
    Options& opt)
{
    cmdline::parser cmd;

    {
        cmd.add<std::string>("input", 'i', "input filename", true);
        cmd.add<std::string>("output", 'o', "output filename base", false);
        cmd.add<std::string>("type", 't', "export type(m = model, a = animation)", true, "m");
        cmd.add<std::string>("base", 'b', "input filename for animation base model", false);
        cmd.add("gpu", 'g', "export for gpu skinning");

        cmd.add("help", '?', "print usage");
    }

    bool isCmdOk = cmd.parse(argc, argv);

    if (cmd.exist("help")) {
        std::cerr << cmd.usage();
        return false;
    }

    if (!isCmdOk) {
        std::cerr << cmd.error_full() << std::endl << cmd.usage();
        return false;
    }

    return true;
}

int main(int argc, char* argv[])
{
    return 1;
}
