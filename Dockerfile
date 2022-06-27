FROM master.garching.cluster.campar.in.tum.de:10443/camp/ubuntu_20.04-python_3.8-tensorflow_2.6-gpu

RUN sudo apt update
RUN sudo apt-get install libopenmpi-dev -y
RUN env MPICC=/usr/bin/mpicc pip install mpi4py
RUN pip install -r requirements.txt