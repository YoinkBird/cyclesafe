NAME=cs_modelmanager
DOCKER_USER=yoinkbird
TAG=${DOCKER_USER}/${NAME}
_target=release
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

dev_dep: ## develop, persist files back to repo. docker build+run with $PWD mounted rw. E.g. in order to edit code from within a running container.
	docker build --tag ${TAG} . && \
		docker run --rm -it -v ${PWD}:/app:rw ${TAG} bash

dev: ## develop, live-update files from repo. docker build+run with $PWD mounted rw
	docker build --tag ${TAG} . && \
		docker run --rm -it ${TAG} bash
#		docker run --rm -it -v ${PWD}:/app:rw ${TAG} bash
_build:
	docker build --tag ${TAG}_${_target} --target ${_target} .
build: ## docker build
	$(MAKE) -e _target=release _build

build_test: ## docker build test image
	$(MAKE) -e _target=test _build

_run:
	docker container rm ${NAME} || true
	docker run --name ${NAME} ${TAG}
run: ## docker run
	$(MAKE) _run

run_test: override _target = test ## docker run test image
run_test:
	$(MAKE) -e TAG=${TAG}_${_target} _run

#TODO-FUTURE: # serve: build run ## build+run image
test: build_test run_test ## build+run test image


# hard-code container and image kill,rm
reset: ## remove generated containers,images and show overview of such
	docker container stop ${NAME} || true
	docker container stop ${NAME}_test || true
	docker container rm ${NAME} || true
	docker container rm ${NAME}_test || true
	docker image rm ${TAG} || true
	docker image rm ${TAG}_release || true
	docker image rm ${TAG}_test || true
	echo "MANUAL REVIEW:"
	docker container ls -a
	docker image ls -a

# dev notes:
# idea adapted from to https://earthly.dev/blog/docker-and-makefiles/
# target specific vars: https://www.gnu.org/software/make/manual/make.html#Target_002dspecific
