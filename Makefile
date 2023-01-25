NAME=cs_modelgen
DOCKER_USER=yoinkbird
TAG=${DOCKER_USER}/${NAME}
# iterate on dependencies; mount rw to automatically update with changes to requirements.txt for pip
dev_dep:
	docker build --tag ${TAG} . && \
		docker run --rm -it -v ${PWD}:/app:rw ${TAG} bash
dev:
	docker build --tag ${TAG} . && \
		docker run --rm -it -v ${PWD}:/app:ro ${TAG} bash
build:
	docker build --tag ${TAG} .

run:
	docker run ${TAG}
