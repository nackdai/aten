# Python development

## Install pyenv

## Prepare venv

```shell
python3 -m venv .venv
```

`.venv` directory is created and it's specified in `.gitignore` to avoid pushing it accidentally.

If we'd like to active the virtual environment:

```shell
source .venv/bin/activate
```

or

```shell
. .venv/bin/activate
```

If we'd like to deactivate it:

```shell
deactivate
```

## Install a project in editable mode

```shell
pip install -e .
```
