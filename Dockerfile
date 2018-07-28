FROM zhai/adw_env
LABEL maintainer="zhai@uchicago.edu"

COPY 		. /app
WORKDIR 	/app

#ENV FLASK_APP = app.py
#ENV FLASK_ENV = production
#ENV FLASK_DEBUG = 0

ENTRYPOINT 	["python"]
CMD 		["app.py"]

#export DOCKER_HOST=tcp://localhost:2375