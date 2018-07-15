FROM 		ubuntu:latest

LABEL maintainer="zhai@uchicago.edu"

RUN			apt-get update -y &&\
			apt-get install -y python3 python3-pip


COPY 		. /app
WORKDIR 	/app
RUN 		pip3 install -r requirements_docker.txt

RUN [ "python3", "-c", "import nltk; nltk.download('punkt')" ]
RUN [ "python3", "-c", "import nltk; nltk.download('wordnet')" ]

ENTRYPOINT 	["python3"]
CMD 		["app.py"]