# Before You Begin: Setting Up Linux with WSL

This course runs TFX, which needs Linux. On Windows you get Linux through Windows Subsystem for Linux (WSL 2) with Ubuntu 22.04, which takes about 20 to 30 minutes to set up.

WSL runs a real Ubuntu system inside Windows, so Linux tools such as TFX work as they would on a Linux machine. Complete this guide before Module 2: every notebook and pipeline in the course assumes this setup, and the labs were recorded in VS Code connected to Ubuntu 22.04 on WSL. You need administrator rights on your Windows computer.

**Why Linux matters beyond this course:** almost every production ML pipeline runs on Linux, including the Vertex AI containers used from Module 7 onward. Working in Linux from day one keeps your local and cloud setups consistent.

## Why Linux, and which path is yours

TFX depends on compiled libraries such as tfx-bsl that are published only for Linux and Intel-based macOS. There is no native Windows build, so `pip install tfx` fails on plain Windows ([tfx-bsl 1.15.1 files on PyPI](https://pypi.org/project/tfx-bsl/1.15.1/#files)).

| Your computer | What to do |
| --- | --- |
| Windows 10 (version 2004, build 19041 or later) or Windows 11 | Follow this guide: install WSL 2 with Ubuntu 22.04 |
| Linux (Ubuntu 22.04 recommended) | Skip to *Set up Python 3.10 and the course environment* |
| macOS on Intel | TFX installs natively; follow the Python steps in your terminal |
| macOS on Apple Silicon (M1 to M4) | No native TFX build exists; use a Linux virtual machine or a cloud Linux VM |

We use Ubuntu 22.04 because it ships Python 3.10, the version the course's TFX 1.15 stack is pinned to.

## Install WSL 2 with Ubuntu 22.04

1. Open the Start menu, type **PowerShell**, right-click it and choose **Run as administrator**.
2. Install WSL with Ubuntu 22.04:

   ```powershell
   wsl --install -d Ubuntu-22.04
   ```

3. Restart your computer when the install finishes.
4. Ubuntu opens on its own after the restart (or open **Ubuntu 22.04** from the Start menu). The first launch takes a minute while files decompress.
5. Create a Linux username and password when asked. The password does not show as you type; that is normal. You will need it for `sudo` commands.
6. Confirm you are on WSL 2. In PowerShell, check that the VERSION column shows 2:

   ```powershell
   wsl --list --verbose
   ```

7. Update Ubuntu's packages from the Ubuntu terminal:

   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

Keep your course files inside Linux (for example `~/projects`), not under `/mnt/c`: Microsoft recommends the Linux file system for speed when you work from the Linux command line. To open your current Linux folder in Windows File Explorer, run `explorer.exe .`

## Set up Python 3.10 and the course environment

1. Install the venv and pip tools:

   ```bash
   sudo apt install -y python3.10-venv python3-pip git
   ```

2. Create and activate a virtual environment:

   ```bash
   python3.10 -m venv ~/tfx_venv
   source ~/tfx_venv/bin/activate
   python -m pip install --upgrade pip
   ```

3. Install the course packages from the repository root:

   ```bash
   pip install -r requirements.txt
   ```

Run `source ~/tfx_venv/bin/activate` each time you open a new terminal.

## Connect VS Code to WSL

1. Install [VS Code](https://code.visualstudio.com/) on Windows.
2. Install the **WSL** extension (publisher: Microsoft), plus **Python** and **Jupyter**.
3. From the Ubuntu terminal, open your project folder in VS Code:

   ```bash
   cd ~/projects/production-grade-dl-pipelines
   code .
   ```

4. The bottom-left corner should show **WSL: Ubuntu-22.04**.
5. Run **Python: Select Interpreter** (Ctrl+Shift+P) and pick `~/tfx_venv/bin/python`. Use the same environment as the notebook kernel.

## Verify your setup

```bash
python --version
python -c "import tfx; print('TFX', tfx.__version__)"
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
python -c "import mlflow; print('MLflow', mlflow.__version__)"
```

- [ ] Python 3.10
- [ ] TFX 1.15.1
- [ ] TensorFlow 2.15.1
- [ ] MLflow prints a version with no error

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Error 0x80370102 or "a required feature is not installed" | Turn on **Virtual Machine Platform** in *Turn Windows features on or off*, enable CPU virtualization in BIOS/UEFI, restart |
| Error 0x8007019e | Enable the WSL component in *Turn Windows features on or off* and restart |
| Install stuck at 0.0% | `wsl --install --web-download -d Ubuntu-22.04` |
| `wsl` is not recognized | Run `wsl.exe` from PowerShell or Command Prompt as administrator |
| VERSION shows 1 | `wsl --set-version Ubuntu-22.04 2` |
| `pip install tfx` fails or picks the wrong Python | Activate the environment and check `python --version` shows 3.10 |
| Notebooks run slowly | Move the project from `/mnt/c/...` into `~/projects` |
| VS Code does not show WSL | Reopen it from the Ubuntu terminal with `code .` |

## Sources

- [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install), Microsoft Learn
- [Troubleshooting WSL](https://learn.microsoft.com/en-us/windows/wsl/troubleshooting), Microsoft Learn
- [Working across Windows and Linux file systems](https://learn.microsoft.com/en-us/windows/wsl/filesystems), Microsoft Learn
- [Developing in WSL](https://code.visualstudio.com/docs/remote/wsl), VS Code docs
