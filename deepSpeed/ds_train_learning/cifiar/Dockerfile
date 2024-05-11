# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

RUN apt-get update
RUN apt-get install -y openmpi-bin 
RUN apt-get install -y libopenmpi-dev
RUN apt-get install -y ninja-build  
RUN apt-get install -y python3-mpi4py
RUN apt-get install -y numactl
RUN echo "Setting System Param >>>>>"
RUN echo "export PATH=/usr/bin/mpirun:\$PATH" >> ~/.bashrc
RUN echo "export PATH=/usr/bin/mpiexec:\$PATH" >> ~/.bashrc
RUN echo "export PATH=/opt/conda/bin/ninja:\$PATH" >> ~/.bashrc
RUN echo "export PATH=/usr/bin/mpirun:\$PATH" >> ~/.profile
RUN echo "export PATH=/usr/bin/mpiexec:\$PATH" >> ~/.profile
RUN echo "export PATH=/opt/conda/bin/ninja:\$PATH" >> ~/.profile

RUN echo 'export NUMA_POLICY=preferred' >> ~/.bashrc
RUN echo 'export NUMA_NODES=0' >> ~/.bashrc

RUN pip3 install deepspeed==0.12.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN conda install -c conda-forge mpi4py
# RUN pip3 install mpi4py  -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install tqdm   -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install triton  -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install tensorboard  -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN echo "Done"

