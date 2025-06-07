from __future__ import annotations

import argparse
import os

from lxml import etree

NAMESPACE_MAP = {"ns": "http://schemas.microsoft.com/developer/msbuild/2003"}


def parse_compile_items_from_item_group(
    item_group_element: etree.Element,
    tag: str,
    attrib: str,
    item_list: list[str],
):
    """Parse compile items from ItemGroup element.

    Args:
        item_group_element: ItemGroup element.
        tag: Tag to compile item element in ItemGroup.
        attrib: Attribute to specifiy compiie item.
        item_list: List to store parsed compile item.
    """
    include_items = item_group_element.findall(f".//ns:{tag}", NAMESPACE_MAP)
    if include_items is not None:
        for item in include_items:
            value = item.get(attrib, None)
            if value is not None:
                item_list.append(value.replace("\\", "/"))


def parse_compile_items(vc_proj_file: str) -> tuple[list[str], list[str]]:
    """Parse compile items from Visual Studio project file.

    Args:
        vc_proj_file: Path to Visual Studio project file.

    Returns:
        Tuple to parsed compile item list and parsed compile item list for CUDA.
    """
    try:
        parser = etree.XMLParser(remove_comments=True)
        xml = etree.parse(vc_proj_file, parser)

        item_groups = xml.findall(".//ns:ItemGroup", NAMESPACE_MAP)

        compile_items: list[str] = []
        cuda_compile_items: list[str] = []

        for group in list(item_groups):
            parse_compile_items_from_item_group(
                group,
                "ClInclude",
                "Include",
                compile_items,
            )
            parse_compile_items_from_item_group(
                group,
                "ClCompile",
                "Include",
                compile_items,
            )
            parse_compile_items_from_item_group(
                group,
                "CudaCompile",
                "Include",
                cuda_compile_items,
            )

        return [compile_items, cuda_compile_items]
    except Exception as err:
        print(f"{err}")
        os.abort()


def format_for_cmake_linux(
    compile_items: list[str],
    workdir: str,
    vc_proj_file: str,
    basepath: str,
):
    """Format item string to fit into CMake.

    Args:
        compile_items: List to store compile item.
        workdir: Path to working directory.
        vc_proj_file: Path to Visual Studio project file.
        basepath: Base path to create relative path.
    """
    try:
        vc_proj_dir = os.path.dirname(vc_proj_file)

        for i in range(len(compile_items)):
            item = compile_items[i]

            # This is only for Linux. In order to highlight Windoes stuff, do nothing.
            if "windows" in item:
                continue

            rel_path = os.path.relpath(
                workdir + "/" + vc_proj_dir + "/" + item,
                workdir + "/" + basepath,
            )

            compile_items[i] = "  " + rel_path
    except Exception as err:
        print(f"{err}")


def trim_end_path_separtor(path: str) -> str:
    """Trim if end of path is separator '/'.

    Args:
        path: Path as string.

    Returns:
        If end of path is separator '/', returns trimed path. Otherwise, returns sepcified path directly.
    """
    if path.endswith("/"):
        path = path.rstrip("/")
        if len(path) == 0:
            path = "."
    return path


def dump_list(list: list[str]):
    """Dump list.

    Args:
        list: List to be dumped.
    """
    for item in list:
        print(f"{item}")


def main():
    # NOTE:
    # e.g.
    # python3 ./tools/extract_file_from_vs_proj.py -v vs2019/libaten.vcxproj -o libaten.txt -b src/libaten  # noqa: E501
    parser = argparse.ArgumentParser(
        description="Extract compile files from vs proj file",
    )
    parser.add_argument(
        "-v",
        "--vcproj",
        type=str,
        help="VC proj file to extract",
        required=True,
        default=None,
    )
    parser.add_argument("-o", "--output", type=str, help="File to output", default=None)
    parser.add_argument(
        "-b",
        "--basepath",
        type=str,
        help="Base path to convert to relative path",
        required=True,
        default=None,
    )
    parser.add_argument(
        "-w",
        "--workdir",
        type=str,
        help="Working directory",
        default=".",
    )
    args = parser.parse_args()

    args.basepath = trim_end_path_separtor(args.basepath)
    args.workdir = trim_end_path_separtor(args.workdir)

    compile_items, cuda_compile_items = parse_compile_items(args.vcproj)

    cpp_as_cuda_list: list[str] = []
    for item in cuda_compile_items:
        if ".cpp" in item or ".cxx" in item:
            cpp_as_cuda_list.append(item)
        compile_items.append(item)

    cuda_compile_items.clear()

    compile_items = sorted(compile_items)
    cpp_as_cuda_list = sorted(cpp_as_cuda_list)

    format_for_cmake_linux(compile_items, args.workdir, args.vcproj, args.basepath)
    format_for_cmake_linux(cpp_as_cuda_list, args.workdir, args.vcproj, args.basepath)

    if args.output is not None:
        with open(args.output, mode="w") as f:
            for item in compile_items:
                f.write(f"{item}\n")
            f.write("CPP AS CUDA ====\n")
            for item in cpp_as_cuda_list:
                f.write(f"{item}\n")
    else:
        dump_list(compile_items)
        print("CPP AS CUDA ====")
        dump_list(cpp_as_cuda_list)


if __name__ == "__main__":
    main()
