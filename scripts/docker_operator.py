from __future__  import annotations

import asyncio
import argparse
import os
import socket
import sys

from asyncio.subprocess import Process
from enum import Enum
from typing import List, Optional

class ProcessRunner:
    """Class to run sub process."""
    def __init__(self):
        self._proc: Process = None

    async def start(
            self, program: str, args: List[str],
            stdout_type: Optional[int], stderr_type: Optional[int]):
        """Invoke sub process.

        Args
            program: Program to be run as sub process.
            args: Arguments to pass to program.
            stdout_type: Stream type for stdout.
            stderr_type: Stream type for stderr.
        """
        self._proc = await asyncio.create_subprocess_exec(
            program,
            *args,
            stdout=stdout_type,
            stderr=stderr_type,
        )

    async def stream(self, interval: float):
        """Returns stream from executing sub process.

        Args:
            interval: Time to interval.

        Yields:
            String from stdoud and string from stderr.
        """
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
        """Wait for sub process finishes."""
        if self._proc is None:
            return None

        await self._proc.communicate()
        return self._proc.returncode

class DockerContainuerRunningMode(Enum):
    """Mode how to run docker container."""
    Detouch = 1
    Enter = 2

async def run_process(
        program: str,  args: List[str],
        stdout_type: Optional[int], stderr_type: Optional[int]) -> Optional[int]:
    """Run process

    Args
        program: Program to be run as sub process.
        args: Arguments to pass to program.
        stdout_type: Stream type for stdout.
        stderr_type: Stream type for stderr.

    Returns:
        Return code from sub process.
    """
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
    """Run docker container.

    Args:
        docker_image: Docker image.
        container_name: Docker container name.
        mode: Mode how to run docker container.

    Returns:
        Return code from docker run command as sub process.
    """
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
    """Exectute command in specified docker container.

    Args:
        container_name: Container name to execute command.
        command: Command to be executed in docker container.

    Returns:
        Return code from docker exec as sub process.
    """
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
    """Kill docker container.

    Args:
        container_name: Container name to be killed.
    """
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
