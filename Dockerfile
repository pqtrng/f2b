FROM tensorflow/tensorflow:latest-devel-gpu AS env

WORKDIR /f2b

COPY requirements.txt .

COPY setup.py .

RUN apt-get update && apt-get upgrade -y

RUN pip install -r requirements.txt

FROM env

COPY . ./

# CMD ["python", "src/train.py"]
