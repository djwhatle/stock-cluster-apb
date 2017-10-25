FROM centos:7

EXPOSE 8080

RUN mkdir -p /usr/src/stock-categorizer
COPY requirements.txt /usr/src/stock-categorizer
WORKDIR /usr/src/stock-categorizer/

RUN yum -y install epel-release
RUN yum -y install python34 python34-pip python34-tkinter
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src /usr/src/stock-categorizer/src
WORKDIR /usr/src/stock-categorizer/src

#USER 99
ENTRYPOINT [ "entrypoint.sh" ]
CMD python app.py
