from __future__ import annotations

import argparse
import asyncio
import os
import socket
import subprocess
import sys
from asyncio.subprocess import Process
from enum import Enum
from typing import List, Optional


class ProcessRunner:
    """Class to run sub process."""

    def __init__(self):
        self._proc: Process = None

    async def start(
        self,
        program: str,
        args: List[str],
        stdout_type: Optional[int],
        stderr_type: Optional[int],
    ):
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
    program: str,
    args: List[str],
    stdout_type: Optional[int],
    stderr_type: Optional[int],
) -> Optional[int]:
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
            print(stdout, end="", flush=True)
        if stderr:
            print(stderr, end="", flush=True, file=sys.stderr)

    returncode = await proc_runner.wait()
    return returncode


async def run_docker_container(
    docker_image: str, container_name: str, mode: DockerContainuerRunningMode, exec_command: Optional[str]
) -> Optional[int]:
    """Run docker container.

    Args:
        docker_image: Docker image.
        container_name: Docker container name.
        mode: Mode how to run docker container.
        exec_command: String for command to execute direclty in docker container.

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
        "run",
        "-it",
        "--rm",
        "-u",
        f"{userid}:{groupid}",
        "-h",
        hostname,
        "-w",
        pwd,
        "--mount",
        f"type=bind,src={pwd},target={pwd}",
        "--mount",
        f"type=bind,src={pwd}/.home,target={home}",
        "-e",
        f"HOME={home}",
        "--mount",
        "type=bind,src=/etc/passwd,target=/etc/passwd,readonly",
        "--mount",
        "type=bind,src=/etc/group,target=/etc/group,readonly",
        "--name",
        container_name,
    ]

    # If command to execute is specified, to display log, container should not be launched as detouch.
    if mode == DockerContainuerRunningMode.Detouch and exec_command is None:
        args.append("-d")

    args.append(docker_image)

    if mode == DockerContainuerRunningMode.Enter:
        args.append("bash")
        stdout_type = None
        stderr_type = None
    elif exec_command is not None:
        args.append("bash")
        args.append("-c")
        args.append(f"{exec_command}")

    returncode = await run_process("docker", args, stdout_type, stderr_type)
    return returncode


async def execute_command_in_docker_container(
    container_name: str, command: Optional[str]
) -> Optional[int]:
    """Exectute command in specified docker container.

    Args:
        container_name: Container name to execute command.
        command: Command to be executed in docker container.

    Returns:
        Return code from docker exec as sub process.
    """
    if command is None:
        return 0

    args = [
        "exec",
        container_name,
        "bash",
        "-c",
    ]

    args.append(command)

    returncode = await run_process(
        "docker", args, asyncio.subprocess.PIPE, asyncio.subprocess.PIPE
    )
    return returncode

def check_if_container_is_running(container_name: Optional[str]) -> bool:
    """Check if container is running.

    Args:
        container_name: Container name to execute command.

    Returns:
        If container is running, returns True. Otherwise, returns False.
    """
    if container_name is None:
        return False

    cmds = [
        "docker", "container", "inspect", "-f", "\'{{.State.Running}}\'", container_name
    ]

    # NOTE:
    # In order to get stdout direclty, use subprocess directly.
    proc = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = proc.stdout.read().decode()
    proc.communicate()

    if proc.returncode != 0:
        return False

    if "true" in result:
        return True

    return False

async def kill_docker_container(container_name: str):
    """Kill docker container.

    Args:
        container_name: Container name to be killed.
    """
    kill_args = ["kill", container_name]
    _ = await run_process(
        "docker", kill_args, asyncio.subprocess.DEVNULL, asyncio.subprocess.STDOUT
    )

    rm_args = ["container", "rm", container_name]
    _ = await run_process(
        "docker", rm_args, asyncio.subprocess.DEVNULL, asyncio.subprocess.STDOUT
    )


async def main(container_name: str):
    # NOTE:
    # e.g.
    # python3 ./scripts/docker_operator.py -i ghcr.io/nackdai/aten/aten_dev:latest -c "pre-commit run -a" -r
    parser = argparse.ArgumentParser(description="Run clang-tidy")
    parser.add_argument("-i", "--image", type=str, help="docker image", default=None)
    parser.add_argument("-n", "--name", type=str, help="container name", default=None)
    parser.add_argument(
        "-e",
        "--enter",
        action="store_true",
        help="Enter docker container",
        default=False,
    )
    parser.add_argument(
        "-r",
        "--remove",
        action="store_true",
        help="Remove docker container",
        default=False,
    )
    parser.add_argument(
        "-c", "--command", type=str, help="Commands to be executed", default=None
    )
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

    returncode = 0

    if args.enter:
        # Kill container forcibly.
        await kill_docker_container(container_name)
        returncode = await run_docker_container(
            args.image, container_name, DockerContainuerRunningMode.Enter
        )
    else:
        returncode = 0

        # If specified container is not running, launch docker image.
        if check_if_container_is_running(container_name):
            print(f"Container \"{container_name}\" is already running")
            if args.command is not None:
                # Specified container is running. So, run docker exec command.
                returncode = await execute_command_in_docker_container(
                    container_name, args.command
                )
        else:
            # Execute command directily with docker run command.
            returncode = await run_docker_container(
                args.image, container_name, DockerContainuerRunningMode.Detouch, args.command
            )

    if args.remove:
        await kill_docker_container(container_name)

    if returncode is None or returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    # NOTE
    # After KeyboardInterrupt (Ctrl+C), RuntimeError is raised.
    # But, it's cpython's issue. It has been fixed at 3.11.1.
    # https://github.com/python/cpython/issues/96827
    container_name = ""
    try:
        asyncio.run(main(container_name))
    except KeyboardInterrupt:
        asyncio.run(kill_docker_container(container_name))
