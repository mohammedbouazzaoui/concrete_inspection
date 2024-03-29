FROM python:3.9-slim-buster
COPY requirements.txt /
RUN pip install --no-cache-dir -r requirements.txt 
RUN pip install -U pip setuptools wheel 
RUN pip install -U spacy 
RUN python -m spacy download nl_core_news_md
RUN pip install Pillow
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install keras
RUN pip install sklearn
RUN pip install scipy
RUN pip install autokeras
RUN pip install tensorflow
RUN pip install opencv-python
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install iputils-ping -y
RUN apt-get install net-tools -y
RUN apt-get install vim -y
COPY ./project /project
COPY ./static  /static 
WORKDIR /project
CMD ["python", "app.py"]
EXPOSE 5000
