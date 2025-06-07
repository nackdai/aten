# Python development

## pyenv

### Linux

We can install `pyenv` as follows:

```bash
curl https://pyenv.run | bash
```

After installation, we will see instructions to add some lines to your `.bashrc`.
To enable `pyenv`, add the following to your `.bashrc`:

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

### Windows

Refer to [this Japanese article](https://qiita.com/probabilityhill/items/9a22f395a1e93206c846).

## venv

We can create a local virtual environment with:

```bash
python3 -m venv .venv
```

The `.venv` directory will be created, and it is already listed in `.gitignore` to prevent
accidental commits.

To activate the virtual environment:

- **Linux**

```bash
source .venv/bin/activate
```

or

```bash
. .venv/bin/activate
```

- **Windows**

We can activate the virtual environment in both PowerShell and cmd with:

```powershell
.\.venv\Scripts\activate
```

To deactivate the virtual environment on any platform:

```bash
deactivate
```

## Install a project in editable mode

We can install the dependencies and your development source code in editable mode so that changes
to the source code are immediately reflected without reinstalling.
This is useful for development, as we will often update the source code.

To install in editable mode:

```bash
pip install -e .
```

Currently, there is no configuration to build the Python module for deployment.

To install optional dependencies, use:

```bash
pip install -e .[dev]
```

If we have several optional dependencies defined, for example:

```toml
[project.optional-dependencies]
dev = ["pytest"]
cli = ["rich", "click"]
```

We can specify multiple keys in brackets:

```bash
pip install -e .[dev,cli]
```
