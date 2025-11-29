PYTHON=python3
VENV=venv
REQUIREMENTS=requirements.txt
KERNEL_NAME=myproject
KERNEL_DISPLAY="My Project Kernel"
all: check-python-version create-venv install-requirements kernel
#1. Check python's version
check-python-version:
	@which $(PYTHON) || (echo "Python not found, install with:sudo pacman -S python" && exit 1)
	@$(PYTHON) --version
#2. Create a virtual environment.	
create-venv:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV) && echo "Virtual environment created at venv"
	@echo "To activate the virtual environment, run: source $(VENV)/bin/activate"
#3. Install packages from requirements.txt
install-requirements: check-python-version create-venv
	. $(VENV)/bin/activate && pip install --upgrade pip
	. $(VENV)/bin/activate && pip install -r $(REQUIREMENTS)
#4. Register jupyter kernel for vs code
kernel:install-requirements
	. $(VENV)/bin/activate && python -m ipykernel install --user --name=$(KERNEL_NAME) --display-name $(KERNEL_DISPLAY)
	@echo "kernel registered. In vscode, select the kernel named $(KERNEL_DISPLAY)"
#5. Clean up the virtual environment.
clean-venv:
	@rm -rf $(VENV) && pip cache purge && echo "Virtual environment is removed."