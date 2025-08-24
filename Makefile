IMAGE_NAME   = rnl-docker-cuda
TAG          = latest
IMAGE        = $(IMAGE_NAME):$(TAG)
VERSION      = 1.1

MODE              ?= learn
MAX_TIMESTEP_GLOBAL ?= 5_000_000
POPULATION 		  ?= 2
LOOP_FEEDBACK 	  ?= 10
DESCRIPTION_TASK  ?= None
SEED              ?= 1
BATCH_SIZE        ?= 64
NUM_ENVS          ?= 16
DEVICE            ?= mps
LEARN_STEP        ?= 256
CHECKPOINT        ?= 10000
CHECKPOINT_PATH   ?= checkpoint_models_custom_banca
LR                ?= 1e-5
GAE_LAMBDA        ?= 0.90
ENT_COEF          ?= 0.05
CLIP_RANGE_VF        ?= 0.2
TARGET_KL          ?= 0.025
VF_COEF           ?= 0.5
MAX_GRAD_NORM     ?= 0.5
UPDATE_EPOCHS     ?= 3
NAME              ?= rnl-real
SCALAR            ?= 20
CONTROL           ?= False
USE_WANDB         ?= True
USE_AGENTS        ?= False
PRETRAINED        ?= None
VERBOSE           ?= True
MAP_NAME          ?= map
TYPE 		  		?= custom
OBSTACLE_PERCENTAGE ?= 0
MAP_SIZE          ?= 0
POLICY       	  ?= PPO

TRAIN_ARGS = \
	$(MODE) \
	--agent $(USE_AGENTS) \
	--max_timestep_global $(MAX_TIMESTEP_GLOBAL) \
	--population $(POPULATION) \
	--loop_feedback $(LOOP_FEEDBACK) \
	--description_task $(DESCRIPTION_TASK) \
	--seed $(SEED) \
	--batch_size $(BATCH_SIZE) \
	--num_envs $(NUM_ENVS) \
	--device $(DEVICE) \
	--checkpoint $(CHECKPOINT) \
	--checkpoint_path $(CHECKPOINT_PATH) \
	--lr $(LR) \
	--learn_step $(LEARN_STEP) \
	--gae_lambda $(GAE_LAMBDA) \
	--ent_coef $(ENT_COEF) \
	--vf_coef $(VF_COEF) \
	--max_grad_norm $(MAX_GRAD_NORM) \
	--update_epochs $(UPDATE_EPOCHS) \
	--clip_range_vf $(CLIP_RANGE_VF) \
	--target_kl $(TARGET_KL) \
	--name $(NAME) \
	--controller False \
	--pretrained $(PRETRAINED) \
	--verbose $(VERBOSE) \
	--debug True \
	--scalar $(SCALAR) \
	--use_wandb $(USE_WANDB) \
	--type $(TYPE) \
 	--obstacle_percentage $(OBSTACLE_PERCENTAGE) \
  	--map_size $(MAP_SIZE) \
   	--policy $(POLICY) \

DOCKER_RUN_COMMON = -e WANDB_API_KEY=$(WANDB_API_KEY) \
	-e GEMINI_API_KEY=$(GEMINI_API_KEY) \
	--gpus all \
	--network host \
	--privileged \
	--memory=16g \
	-it \
	-v $(PWD):/workdir \
	$(IMAGE_NAME)

define PRINT_CONFIG
	@echo "MODE=$(MODE)"
	@echo "MAX_TIMESTEP_GLOBAL=$(MAX_TIMESTEP_GLOBAL)"
	@echo "POPULATION=$(POPULATION)"
	@echo "LOOP_FEEDBACK=$(LOOP_FEEDBACK)"
	@echo "DESCRIPTION_TASK=$(DESCRIPTION_TASK)"
	@echo "SEED=$(SEED)"
	@echo "BATCH_SIZE=$(BATCH_SIZE)"
	@echo "NUM_ENVS=$(NUM_ENVS)"
	@echo "DEVICE=$(DEVICE)"
	@echo "CHECKPOINT=$(CHECKPOINT)"
	@echo "LR=$(LR)"
	@echo "GAE_LAMBDA=$(GAE_LAMBDA)"
	@echo "ENT_COEF=$(ENT_COEF)"
	@echo "VF_COEF=$(VF_COEF)"
	@echo "MAX_GRAD_NORM=$(MAX_GRAD_NORM)"
	@echo "UPDATE_EPOCHS=$(UPDATE_EPOCHS)"
	@echo "CLIP_RANGE_VF=$(CLIP_RANGE_VF)"
	@echo "TARGET_KL=$(TARGET_KL)"
	@echo "NAME=$(NAME)"
	@echo "WANDB_API_KEY=$(WANDB_API_KEY)"
	@echo "GEMINI_API_KEY=$(GEMINI_API_KEY)"
	@echo "USE_WANDB=$(USE_WANDB)"
	@echo "SCALAR=$(SCALAR)"
	@echo "CONTROL=$(CONTROL)"
	@echo "USE_AGENTS=$(USE_AGENTS)"
	@echo "PRETRAINED=$(PRETRAINED)"
	@echo "VERBOSE=$(VERBOSE)"
	@echo "TYPE=$(TYPE)"
	@echo "OBSTACLE_PERCENTAGE=$(OBSTACLE_PERCENTAGE)"
	@echo "MAP_SIZE=$(MAP_SIZE)"
	@echo "POLICY=$(POLICY)"
	@echo
endef

.PHONY: sim
sim:
	@uv run python -m main sim

.PHONY: learn
learn:
	@uv run python -m main $(TRAIN_ARGS)

.PHONY: training
training:
	@uv run python rnl/training/train.py

.PHONY: probe
probe:
	@uv run python -m main run \
		--num_envs $(NUM_ENVS) \
		--device $(DEVICE) \
		--max_timestep_global $(MAX_TIMESTEP_GLOBAL) \
		--controller False \
		--type $(TYPE) \
		--seed $(SEED) \
		--debug True \
		--scalar $(SCALAR)

.PHONY: probe-spawn
probe-spawn:
	@uv run python rnl/training/probes.py

.PHONY: test_without_coverage
test_without_coverage:
	@uv run pytest -s -x -vv -p no:warnings

.PHONY: gazebo
gazebo:
	@uv run python rnl/engine/generate.py --folder ./data/$(MAP_NAME) --name $(MAP_NAME)

.PHONY: test
test:
	@uv run pytest -s -x --cov=rnl -vv -p no:warnings

.PHONY: plot_test
plot_test:
	@uv run python tests/test_reward.py

.PHONY: post_test
post_test:
	@uv run coverage html

.PHONY: publish
publish:
	@uv publish --build -u __token__ -p $(RNL_PYPI_TOKEN)

.PHONY: install_with_dev
install_with_dev:
	@uv sync

.PHONY: install
install:
	@uv sync --no-group dev

.PHONY: build
build:
	@docker build -f docker/Dockerfile -t $(IMAGE_NAME) .

.PHONY: visualize
visualize:
	@uv run rnl/training/visualize.py --save output_map.rrd

.PHONY: train
train:
	@echo
	$(PRINT_CONFIG)
	@docker run -d $(DOCKER_RUN_COMMON) $(TRAIN_ARGS)

.PHONY: train-without
train-without:
	@echo
	$(PRINT_CONFIG)
	@docker run $(DOCKER_RUN_COMMON) $(TRAIN_ARGS)

.PHONY: clean
clean:
	@sudo docker image prune -f

.PHONY: push
push:
	@docker tag $(IMAGE_NAME):$(VERSION) $(IMAGE_NAME):latest
	@docker push $(IMAGE_NAME):latest

uv     = uv
BLACK  = $(uv) run black
ISORT  = $(uv) run isort
RUFF   = $(uv) run ruff
SRC_DIR= ./

.PHONY: lint
lint:
	$(ISORT) $(SRC_DIR)
	$(BLACK) $(SRC_DIR)
	$(RUFF) check $(SRC_DIR)
