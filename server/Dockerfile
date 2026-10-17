# Python 3.12 on Debian. essentia-tensorflow only has x86_64 Linux wheels,
# so docker-compose pins this image to linux/amd64.
FROM python:3.12-slim

# ffmpeg: decodes mp3/m4a. build-essential + git: Madmom compiles from source.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg build-essential git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements-build.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-build.txt \
 && pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Fail the build now, not at first request, if Madmom or Essentia is broken
RUN python -c "import numpy as np; [setattr(np,n,t) for n,t in {'float':float,'int':int,'complex':complex,'object':object,'bool':bool,'str':str}.items() if not hasattr(np,n)]; import collections, collections.abc; [setattr(collections,a,getattr(collections.abc,a)) for a in ('MutableSequence','Iterable','Callable','MutableMapping','Mapping','Sequence') if not hasattr(collections,a)]; from madmom.features.beats import RNNBeatProcessor; import essentia.standard; import faster_whisper; print('imports ok')"

COPY app ./app

# Model files persist in a volume so they download once
ENV MODEL_DIR=/models/essentia \
    HF_HOME=/models/huggingface \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
