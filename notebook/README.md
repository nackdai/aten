# Jupyte Nodebook

## Prerequisite

If the envorimnet is vscode, recommend to install `Jupyter` extention.

Need to prepare `venv`. Call the following in the root directory.

```sh
python -m venv notebook
```

And then, activate venv.

```sh
# Linux
cd notebook
source ./bin/activate

# Windows
cd notebook
Scripts\activate.bat
```

After activating, Install necessary modules via `pip` and `requirements.txt` in this directory.

```sh
# Linux
pip install -r requirements.txt

# Windows
pip install -r requirements_windows.txt
```

If need to deactivate venv, run the following command:

```sh
deactivate
```
