FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy handler and the local modules it imports
COPY handler.py /handler.py
COPY text_normalizer.py /text_normalizer.py
COPY phonetic_engine.py /phonetic_engine.py
COPY audio_stitcher.py /audio_stitcher.py
COPY restore_uzbek_orthography.py /restore_uzbek_orthography.py
COPY cyrillic_to_latin.py /cyrillic_to_latin.py

CMD ["python", "-u", "/handler.py"]
