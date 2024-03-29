#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = f2b
PYTHON_INTERPRETER = python3
TRAINING_TYPE = top
OUTPUT_NETWORK_TYPE = current
DATASET = original
SMOKE_TEST = False

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif


#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: requirements
## Install Python Dependencies
requirements: test_environment
	$(info Install supporting library!)
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

.PHONY: lint
## Format python script
lint:
	$(info format codes)
	autoflake --in-place --remove-unused-variables --remove-all-unused-imports --remove-duplicate-keys *.py
	black .

.PHONY: githook
## Prepare pre commit hooks
githook: lint
	$(info Set up pre commit hook)
	rm .git/hooks/*
	pre-commit install

.PHONY: check
## Check all files before commit
check: githook
	pre-commit run --all-files
	git add .

.PHONY: test
## Test code
test: check
	$(info Testing)
	nosetests

.PHONY: clean
## Clean up binary files and etc
clean: test
	$(info Clean project)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "__MACOSX" -exec rm -rf {} +
	find . -type f -name "*.zip" -delete

.PHONY: plit
## Process and split data
split: clean
	$(info Split dataset!)
	$(PYTHON_INTERPRETER) src/split.py interim processed $(DATASET)

.PHONY: wandb
wandb: clean
	docker pull wandb/local
	docker stop wandb-local || true
	wandb local

.PHONY: train
## Train model
train: split
	$(info Train model)
	$(PYTHON_INTERPRETER) src/train.py $(TRAINING_TYPE) $(DATASET) $(OUTPUT_NETWORK_TYPE) $(SMOKE_TEST)

.PHONY: evaluate
## Evaluate model
evaluate: train
	$(info Evaluate model)
	$(PYTHON_INTERPRETER) src/evaluate.py $(TRAINING_TYPE) $(DATASET) $(OUTPUT_NETWORK_TYPE)


.PHONY: demo
## Evaluate model
demo: clean
	$(info Evaluate model)
	$(PYTHON_INTERPRETER) src/demo.py $(TRAINING_TYPE) $(DATASET) $(OUTPUT_NETWORK_TYPE)

.PHONY: crontab
## Schedule cron jobs
crontab: clean
	$(info Run scheduled jobs)

.PHONY: docker
## Running in a Docker container
docker: clean
	$(info Run in Docker)
	docker build . -t f2b --rm
	docker stop f2b || true && docker rm f2b || true
	docker run --name "f2b" --gpus all -it f2b bash


.PHONY: create_environment
## Create an isolated environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda env remove --name $(PROJECT_NAME)
	conda create --name $(PROJECT_NAME) python=3.8 -y
else
	conda create --name $(PROJECT_NAME) python=2 -y
endif
		@echo ">>> New conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif


.PHONY: test_environment
## Test if the environment exists or not
test_environment:
	$(info Check python version!)
	$(PYTHON_INTERPRETER) environment_test.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
