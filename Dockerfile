FROM python:3.8

WORKDIR /tmp

COPY . /tmp

RUN python -m venv myenv
RUN . myenv/bin/activate

RUN pip3 install -r requirements2.txt
RUN python3 interaction_network/train.py

CMD ["python", "interaction_network/train.py"]
