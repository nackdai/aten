#include "ImageLoader.h"
#include "utility.h"

namespace aten {
    std::string ImageLoader::base_path;

    void ImageLoader::setBasePath(const std::string& base)
    {
        base_path = removeTailPathSeparator(base);
    }

    std::shared_ptr<texture> ImageLoader::load(
        const std::string& path,
        context& ctxt)
    {
        std::string pathname;
        std::string extname;
        std::string filename;

        getStringsFromPath(
            path,
            pathname,
            extname,
            filename);

        return load(filename, path, ctxt);
    }

    std::shared_ptr<texture> ImageLoader::load(
        const std::string& tag,
        const std::string& path,
        context& ctxt)
    {
        std::string fullpath = path;
        if (!base_path.empty()) {
            fullpath = base_path + "/" + fullpath;
        }

        return Image::Load(tag, fullpath, ctxt);
    }
}
