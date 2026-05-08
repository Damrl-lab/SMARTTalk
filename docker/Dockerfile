FROM python:3.10-slim

WORKDIR /artifact

COPY docker/requirements.txt /artifact/docker/requirements.txt
RUN pip install --no-cache-dir -r /artifact/docker/requirements.txt

COPY . /artifact

ENV PYTHONPATH=/artifact

CMD ["bash", "scripts/07_reproduce/reproduce_quick.sh"]
