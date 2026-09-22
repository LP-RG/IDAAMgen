PY := python3.11
ENV_NAME := .venv
DELETION_DELAY ?= 10

# mqbench
MQBENCH_DIR := MQBench
MQBENCH_REPO := git@github.com:ModelTC/MQBench.git

# computed

ACTIV_ENV := . $(ENV_NAME)/bin/activate
IN_ENV := $(ACTIV_ENV) &&

py_init:
	@echo "\n[[ creating python environment if absent ]]"
	$(PY) -m venv $(ENV_NAME)

py_dep: py_init
	@echo "\n[[ installing/upgrading python dependencies ]]"
	$(IN_ENV) python3 -m pip install --upgrade pip setuptools wheel
	$(IN_ENV) pip install --upgrade -r requirements.txt
	$(IN_ENV) pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
	$(IN_ENV) pip install --no-build-isolation ./src/extension_cpp/.

$(MQBENCH_DIR)/.git:
	@echo "\n[[ cloning MQBench ]]"
	git clone $(MQBENCH_REPO) $(MQBENCH_DIR)

mqbench: py_init $(MQBENCH_DIR)/.git
	@echo "\n[[ installing MQBench ]]"
	$(IN_ENV) cd $(MQBENCH_DIR) && pip install -v -e .

setup: py_init py_dep mqbench

rm_cache:
	@echo "\n[[ removing all pycache folders ]]"
	find . -name __pycache__ -prune -print -exec rm -rf {} \;

rm_temp:
	@echo "\n[[ removing all temporary files ]]"
	find . -name "*.tmp" -print -exec rm -rf {} \;

rm_pyenv:
	@echo "\n[[ removing the virtual python environment ]]"
	rm -rf $(ENV_NAME)

rm_mqbench:
	@echo "\n[[ removing MQBench ]]"
	rm -rf $(MQBENCH_DIR)

clean: rm_cache rm_temp rm_pyenv rm_mqbench
	@echo "\n[[ all clean! ]]"
