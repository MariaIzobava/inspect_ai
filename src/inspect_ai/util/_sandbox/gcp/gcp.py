"""Sandbox environment for running code in an (existing) VM on GCP.

On initialisation, creates an SSH multiplexed master connection to the VM.
Then on each call to exec(), the same connection is reused.
This is much faster than SSHing afresh every command.

The following env variables must be specified:
  GCP_USERNAME -- GCP username
  VMS_HOSTNAMES -- a list of available GCP vms (their external IPs) in the format
                   of json list: '["EXTERNAL_IP_1", "EXTERNAL_IP_2", ..."EXTERNAL_IP_N"]'
"""

import asyncio
import json
import logging as logginglib
import os
import pathlib
import shlex
import tempfile
from typing import Literal, Union, overload

from typing_extensions import override

from inspect_ai.util import _subprocess
from inspect_ai.util._sandbox import environment, registry

logging = logginglib.getLogger(__name__)

# List of VMs available for use. Each new sandbox instance pops one off the end.
# Assumes there are more VMs available than the number of sandboxes instances
# created (e.g. the number of epochs). Ideally we would do something like choose
# the VM based on the epoch directly, but doesn't seem to be a way to do that.
_VMS = json.loads(os.environ.get("VMS_HOSTNAMES", "[]"))

# SSH flags used for every command.
_USER = os.environ.get("GCP_USERNAME", "")
_HOME = pathlib.Path.home()
_CONTROL_PATH = _HOME / ".ssh" / "control" / "%C"
_COMMON_ARGS = [
    "-o",
    "ControlMaster=auto",
    "-o",
    f"ControlPath={_CONTROL_PATH}",
    # These silence all the warnings about host keys.
    "-o",
    "LogLevel=ERROR",
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
]


async def _run(*args: str) -> tuple[asyncio.subprocess.Process, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.wait()
    stdout, stderr = await proc.communicate()
    # For some reason, create_subprocess_exec doesn't allow text=True.
    return proc, stdout.decode(), stderr.decode()


def _raise_if_in_tmux() -> None:
    if os.environ.get("TMUX"):
        raise RuntimeError(
            "The GCP sandbox environment doesn't work in tmux. "
            "(The GCP authentication uses the SSH agent which doesn't seem to work "
            "in tmux.) Please run outside of tmux."
        )


@registry.sandboxenv(name="gcp")
class GcpSandboxEnvironment(environment.SandboxEnvironment):
    """Sandbox environment for running code in an (existing) VM on GCP."""

    @override
    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: environment.SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, environment.SandboxEnvironment]:
        _raise_if_in_tmux()
        vm = _VMS.pop()
        logging.debug("Establishing master connection to %s...", vm)
        user_and_host = f"{_USER}@{vm}"
        _CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
        command = ["ssh", "-Nf", *_COMMON_ARGS, user_and_host]

        # We don't use _run() here because if stdout is connected then it hangs.
        # Maybe stdout stays connected even after fork or something...?
        proc = await asyncio.create_subprocess_exec(*command)
        await proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to establish master connection to {vm}. "
                f"SSH had return code {proc.returncode}.\n"
                f"Command was:\n{' '.join(command)}"
            )

        logging.debug("Master connection to %s established!", vm)
        return {"default": GcpSandboxEnvironment(vm, user_and_host)}

    def __init__(self, vm: str, user_and_host: str) -> None:
        super().__init__()
        self._vm = vm
        self._user_and_host = user_and_host

    @override
    async def exec(  # pylint: disable=dangerous-default-value
        self,
        cmd: list[str],
        input: str | bytes | None = None,  # pylint: disable=redefined-builtin
        cwd: str | None = None,
        env: dict[str, str] = {},
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = False,
    ) -> _subprocess.ExecResult[str]:
        """Execute a command within a sandbox environment.

        This implementation only supports the `cmd` and `timeout` args.

        Args:
          cmd: Command or command and arguments to execute.
          input: Standard input. Ignored.
          cwd: Current working dir. Ignored.
          env: Environment variables for execution. Ignored.
          user: Optional username or UID to run the command as. Ignored.
          timeout: Optional execution timeout (seconds).
          timeout_retry: Optional, not used in this implementation.

        Returns:
          Execution result (status code, stderr/stdout, etc.)

        Raises:
          TimeoutError: If the specified `timeout` expires.
        """
        logging.debug("Executing command on %s: %s", self._vm, cmd)
        if input or cwd or env or user:
            raise ValueError(
                "Unsupported arguments for exec in GCP sandbox: "
                f"{input=}, {cwd=}, {env=}, {user=}"
            )
        # We need to shlex.quote here because if we do
        #   ssh foo.com bash --login -c 'echo foo'
        # then SSH blindly concatenates all the arguments into a string with spaces,
        # so the final command is
        #   bash --login -c echo foo
        # Also, the '-n' here is important - otherwise ssh polls stdin, which
        # slows down the asyncio event loop a LOT, even when run async.
        async with asyncio.timeout(timeout):
            proc, stdout, stderr = await _run(
                "ssh",
                "-n",
                *_COMMON_ARGS,
                self._user_and_host,
                *[shlex.quote(arg) for arg in cmd],
            )
        logging.debug("Done executing command on %s: %s", self._vm, cmd)
        return _subprocess.ExecResult[str](
            success=proc.returncode == 0,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    @override
    async def read_file(self, file: str, text: bool = True) -> Union[str, bytes]:
        logging.debug("Reading file %s from %s", file, self._vm)
        result = await self.exec(["cat", file])
        logging.debug("Read file %s from %s", file, self._vm)
        if not result.success:
            raise NotImplementedError(
                "Read file failure handling not implemented.\n"
                f"File: {file!r}\n"
                f"stdout: {result.stdout!r}\n"
                f"stderr: {result.stderr!r}"
            )
        return result.stdout

    @override
    async def write_file(self, file: str, contents: str | bytes) -> None:
        logging.debug("Writing file %s to %s", file, self._vm)
        with tempfile.NamedTemporaryFile() as tmpfile:
            if isinstance(contents, str):
                contents = contents.encode("utf-8")
            tmpfile.write(contents)
            tmpfile.flush()
            proc, stdout, stderr = await _run(
                "scp", *_COMMON_ARGS, tmpfile.name, f"{self._user_and_host}:{file}"
            )
        logging.debug("Wrote file %s to %s", file, self._vm)
        if proc.returncode != 0:
            raise NotImplementedError(
                "Write file failure handling not implemented.\n"
                f"File: {file!r}\n"
                f"stdout: {stdout!r}\n"
                f"stderr: {stderr!r}"
                f"Contents: {file!r}\n"
            )

    # Not implemented.
    @override
    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: environment.SandboxEnvironmentConfigType | None,
        environments: dict[str, environment.SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        pass
