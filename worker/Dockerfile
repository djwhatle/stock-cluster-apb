FROM centos:7

RUN mkdir -p /usr/src/stock-cluster
COPY requirements.txt /usr/src/stock-cluster
WORKDIR /usr/src/stock-cluster/

RUN yum -y install epel-release
RUN yum -y install python34 python34-pip python34-tkinter
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src /usr/src/stock-cluster/src
WORKDIR /usr/src/stock-cluster/src

ENTRYPOINT [ "entrypoint.sh" ]
CMD python3 worker.py
