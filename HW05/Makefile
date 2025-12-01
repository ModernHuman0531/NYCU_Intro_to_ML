PYTHON=python3
VENV=venv
REQUIREMENTS=requirements.txt

all: check-python create-venv install-requirements
# Check python's version
check-python:
	@which $(PYTHON) > /dev/null || (echo "Python3 is not installed. Please install Python3." && exit 1)
	@$(PYTHON) --version

create-venv:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV) && echo "Virtual environment created at $(VENV)"
	@echo "To activate the virtual environment, run: source $(VENV)/bin/activate"

install-requirements: 
	. $(VENV)/bin/activate && pip install --upgrade pip
	. $(VENV)/bin/activate && pip install -r $(REQUIREMENTS)

# Clean up the virtual environment
clean:
	@pip cache purge
	@rm -rf $(VENV) && echo "Virtual environment $(VENV) removed
