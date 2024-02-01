from ubuntu:20.04

RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && apt-get -y install python3 python3.8-venv build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev

COPY . .

WORKDIR "/UAV_Reach_Avoid"
RUN python3 -m venv env
RUN . env/bin/activate && pip install -r requirements.txt

WORKDIR "/Minigrid"
RUN python3 -m venv env
RUN . env/bin/activate && pip install -r requirements.txt && pip install -e gym_minigrid

WORKDIR "/Minigrid/Minigrid2PRISM"
RUN mkdir -p build
WORKDIR "/Minigrid/Minigrid2PRISM/build"
RUN cmake ..
RUN make


WORKDIR "/tempest-devel"
RUN mkdir -p build
WORKDIR "/tempest-devel/build"
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DSTORM_DEVELOPER=OFF -DSTORM_LOG_DISABLE_DEBUG=ON -DSTORM_PORTABLE=ON -DSTORM_USE_SPOT_SHIPPED=ON
RUN make storm-main -j6


ARG TEMPEST_BINARY="/tempest-devel/build/bin/storm"
RUN apt-get -y install vim 
WORKDIR "/"
