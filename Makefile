NAME=cs_modelgen
DOCKER_USER=yoinkbird
TAG=${DOCKER_USER}/${NAME}
_target=release
# dev note: referring to https://earthly.dev/blog/docker-and-makefiles/
# iterate on dependencies; mount rw to automatically update with changes to requirements.txt for pip
dev_dep:
	docker build --tag ${TAG} . && \
		docker run --rm -it -v ${PWD}:/app:rw ${TAG} bash
dev:
	docker build --tag ${TAG} . && \
		docker run --rm -it ${TAG} bash
#		docker run --rm -it -v ${PWD}:/app:rw ${TAG} bash
_build:
	docker build --tag ${TAG}_${_target} --target ${_target} .
build:
	$(MAKE) -e _target=release _build

build_test:
	$(MAKE) -e _target=test _build

_run:
	docker run --name ${NAME} ${TAG}
run:
	$(MAKE) _run

run_test: override _target = test
run_test:
	$(MAKE) -e TAG=${TAG}_${_target} _run

#TODO-FUTURE: # serve: build run
test: build_test run_test
# notes:
# target specific vars: https://www.gnu.org/software/make/manual/make.html#Target_002dspecific


# hard-code container and image kill,rm
reset:
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
