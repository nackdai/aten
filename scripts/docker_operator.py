from __future__  import annotations

import asyncio
import argparse
import os
import socket
import sys

from asyncio.subprocess import Process
from enum import Enum
from typing import List, Optional

# https://www.lifewithpython.com/2021/12/python-subprocess-stream.html
class ProcessRunner:
    def __init__(self):
        self._proc: Process = None

    async def start(
            self, program: str, args: List[str],
            stdout_type: Optional[int], stderr_type: Optional[int]):
        self._proc = await asyncio.create_subprocess_exec(
            program,
            *args,
            stdout=stdout_type,
            stderr=stderr_type,
        )

    async def stream(self, interval: float):
        while True:
            if self._proc.stdout is None or self._proc.stderr is None:
                break

            if self._proc.stdout.at_eof() and self._proc.stderr.at_eof():
                break

            stdout = await self._proc.stdout.readline()
            stderr = await self._proc.stderr.readline()
            yield stdout.decode(), stderr.decode()

            await asyncio.sleep(interval)

    async def wait(self):
        if self._proc is None:
            return None

        await self._proc.communicate()
        return self._proc.returncode

class DockerContainuerRunningMode(Enum):
    Detouch = 1
    Enter = 2

async def run_process(
        program: str,  args: List[str],
        stdout_type: Optional[int], stderr_type: Optional[int]) -> Optional[int]:
    proc_runner = ProcessRunner()

    await proc_runner.start(program, args, stdout_type, stderr_type)

    async for stdout, stderr in proc_runner.stream(0.0):
        if stdout:
            print(stdout, end='', flush=True)
        if stderr:
            print(stderr, end='', flush=True, file=sys.stderr)

    returncode = await proc_runner.wait()
    return returncode

async def run_docker_container(docker_image: str, container_name: str, mode: DockerContainuerRunningMode) -> Optional[int]:
    hostname = socket.gethostname()
    userid = os.getuid()
    groupid = os.getgid()
    pwd = os.getcwd()
    home = os.environ.get("HOME")

    stdout_type = asyncio.subprocess.PIPE
    stderr_type = asyncio.subprocess.PIPE

    args = [
        "run", "-it", "--rm",
        "-u", f"{userid}:{groupid}",
        "-h", hostname,
        "-w", pwd,
        "--mount", f"type=bind,src={pwd},target={pwd}",
        "--mount", f"type=bind,src={pwd}/.home,target={home}",
        "-e", f"HOME={home}",
        "--mount", "type=bind,src=/etc/passwd,target=/etc/passwd,readonly",
        "--mount", "type=bind,src=/etc/group,target=/etc/group,readonly",
        "--name", container_name,
    ]

    if mode == DockerContainuerRunningMode.Detouch:
        args.append("-d")

    args.append(docker_image)

    if mode == DockerContainuerRunningMode.Enter:
        args.append("bash")
        stdout_type = None
        stderr_type = None

    returncode = await run_process("docker", args, stdout_type, stderr_type)
    return returncode

async def execute_command_in_docker_container(container_name: str, command: str) -> Optional[int]:
    args = [
        "exec",
        container_name,
        "bash", "-c",
    ]

    if command:
        args.append(command)

    returncode = await run_process("docker", args, asyncio.subprocess.PIPE, asyncio.subprocess.PIPE)
    return returncode

async def kill_docker_container(container_name: str):
    kill_args = ["kill", container_name]
    _ = await run_process("docker", kill_args, asyncio.subprocess.DEVNULL, asyncio.subprocess.STDOUT)

    rm_args = ["container", "rm", container_name]
    _ = await run_process("docker", rm_args, asyncio.subprocess.DEVNULL, asyncio.subprocess.STDOUT)

async def main():
    # NOTE:
    # e.g.
    # python3 ./scripts/docker_operator.py -i ghcr.io/nackdai/aten/aten_dev:latest -c "pre-commit run -a" -r
    parser = argparse.ArgumentParser(description="Run clang-tidy")
    parser.add_argument('-i', '--image', type=str, help="docker image", required=True, default=None)
    parser.add_argument('-n', '--name', type=str, help="container name", default=None)
    parser.add_argument("-e", "--enter", action="store_true", help="Enter docker container", default=False)
    parser.add_argument("-r", "--remove", action="store_true", help="Remove docker container", default=False)
    parser.add_argument("-c", "--command", type=str, help="Commands to be executed", default=None)
    args = parser.parse_args()

    container_name = args.name

    # If container name is not specified, generate container name from docker image name.
    # e.g. docker image is "a/b/c:latest" -> container name is "c"
    if container_name is None:
        elements = args.image.split("/")
        container_name = elements[-1]
        elements = container_name.split(":")
        if len(elements) == 2:
            container_name = elements[-2]

    if args.remove:
        await kill_docker_container(container_name)

    returncode = 0

    if args.enter:
        returncode = await run_docker_container(args.image, container_name, DockerContainuerRunningMode.Enter)
    else:
        returncode = await run_docker_container(args.image, container_name, DockerContainuerRunningMode.Detouch)
        if returncode is not None and returncode == 0:
            returncode = await execute_command_in_docker_container(container_name, args.command)

    if args.remove:
        await kill_docker_container(container_name)

    if returncode is None or returncode != 0:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
