FROM datamachines/cudnn_tensorflow_opencv:11.3.0_2.5.0_4.5.2-20210601 AS env
WORKDIR /f2b

COPY requirements.txt .

COPY setup.py .

COPY environment_test.py .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

FROM env

COPY . ./

CMD ["nvidia-smi"]
