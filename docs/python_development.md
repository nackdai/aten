# Python development

## pyenv

### Linux

We can install `pyenv` like the following:

```bash
curl https://pyenv.run | bash
```

Then, we can see the log to put some codes in `.bashrc`. To enable pyenv, we need to add the
followings in `.bashrc`:

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

### Windows

Refer the [this (Japanese article)](https://qiita.com/probabilityhill/items/9a22f395a1e93206c846).

## venv

We can create the virtual environment locally with the following:

```bash
python3 -m venv .venv
```

`.venv` directory is created and it has been already specified in `.gitignore` to avoid pushing it accidentally.

If we'd like to active the virtual environment with the following:

- Linux

```bash
source .venv/bin/activate
```

or

```bash
. .venv/bin/activate
```

- Windows

Regardless of PowerShell or cmd, we can activate the virtual environment with the following:

```powershell
.\.venv\Scripts\activate
```

Regardless of Linux or Windows, if we'd like to deactivate:

```bash
deactivate
```

## Install a project in editable mode

We can install the dependencies. We would like to install the our development source codes so that
we can import it from other codes. But, the development source codes are under development. It
means that we would update them so often. If they are installed the built fixed codes as the
module, it's not flexible and inefficient to develop.

Thus, we'd like to install our development source code as the editable mode (i.e. develop mode)
like the following:

```bash
pip install -e .
```

Currently, there is no setting to build the python module to deploy.

If we'e like to install the optional dependencies, we can install them like the following:

```bash
pip install .[dev]
```

Or, for editable mode:

```bash
pip install -e .[dev]
```

If the several optional dependencies are defined like the following:

```toml
[project.optional-dependencies]
dev = ["pytest"]
cli = ["rich", "click",]
```

We can specify the key in `[]`, like the following:

```bash
pip install .[dev, cli]
```

Or, for editable mode:

```bash
pip install -e .[dev, cli]
```
