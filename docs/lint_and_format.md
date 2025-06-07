# Lint and Format

To keep the code clean and consistent, linters and formatters are necessary.

## pre-commit

### What pre-commit is

For more details, see [pre-commit](https://pre-commit.com/).

### How to install pre-commit

We have already defined `pre-commit` as one of the optional dependencies in [.pre-commit-config.yaml](../.pre-commit-config.yaml).
Thus, if you install the dependencies as described in [Install a project in editable mode](python_development.md#install-a-project-in-editable-mode),
`pre-commit` will be installed.

#### Linux

You need to install [shfmt](https://github.com/mvdan/sh) and [shellcheck](https://github.com/koalaman/shellcheck)
locally.

- `shfmt`

```bash
sudo curl -L https://github.com/mvdan/sh/releases/download/v${SHFMT_VERSION}/shfmt_v${SHFMT_VERSION}_linux_amd64 -o /usr/local/bin/shfmt
sudo chmod +x /usr/local/bin/shfmt
```

- `shellcheck`

```bash
sudo apt-get install -y shellcheck
```

#### Windows

On Windows, the following tools do not work stably:

- shfmt
  - This executable works only on Linux. We do not modify shell scripts during native Windows development.
- shellcheck
  - This executable works only on Linux. We do not modify shell scripts during native Windows development.
- cmake-format
  - Theoretically, this should work, but dependencies or path issues may make it unstable
    On native Windows, we mainly develop code in Visual Studio, so we generally do not touch CMake.
- cmake-lint
  - Theoretically, this should work, but dependencies or path issues may make it unstable.
    On native Windows, we mainly develop code in Visual Studio, so we generally do not touch CMake.

### How to run pre-commit

If you follow [Python development](python_development.md), the development environment is created
in `.venv`.
Thus, you need to activate it as described in [venv](python_development.md#venv).

You can run pre-commit as follows:

- **Linux**

```bash
pre-commit run -a
```

- **Windows**

**NOTE:** `_` is used instead of `-` in typing of `pre_commit`.

```bash
python -m pre_commit run -a
```

The reason for the different commands is that the handling of `PATH` and executables differs
between Windows and Linux.

### How to configure pre-commit

We already have `.pre-commit-config.yaml` as the configuration file for `pre-commit`.
For more details, see [pre-commit](https://pre-commit.com/).

## Ruff

### What Ruff is

`Ruff` is a Python linter and code formatter. For more details, see [Ruff](https://docs.astral.sh/ruff/).

### How to install Ruff

We have already defined `ruff` as one of the optional dependencies
in [.pre-commit-config.yaml](../.pre-commit-config.yaml).
Thus, if you install the dependencies as described
in [Install a project in editable mode](python_development.md#install-a-project-in-editable-mode),
`ruff` will be installed.

### How to run Ruff

The procedure is the same as [How to run pre-commit](#how-to-run-pre-commit).
If you follow [Python development](python_development.md), the development environment is created
in `.venv`.
Thus, you need to activate it as described in [venv](python_development.md#venv).

You can run ruff as follows:

- **Linux**

```bash
ruff check
```

- **Windows**

```bash
python -m ruff check
```

The reason for the different commands is that the handling of `PATH` and executables differs
between Windows and Linux.

### How to configure Ruff

We have already defined some configurations in [pyproject.toml](../pyproject.toml).
For more details, see [Ruff configuration](https://docs.astral.sh/ruff/configuration/).
