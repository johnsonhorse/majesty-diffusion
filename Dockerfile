# Add a new comment to trigger build.
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata build-essential wget git git-lfs \
    ffmpeg libsm6 libxext6 \
    && apt-get clean

RUN mkdir -p /src
WORKDIR /src

RUN git clone https://github.com/multimodalart/latent-diffusion --branch 1.6
RUN git clone https://github.com/CompVis/taming-transformers
RUN git clone https://github.com/TencentARC/GFPGAN
RUN git lfs clone https://github.com/LAION-AI/aesthetic-predictor

RUN pip install tensorflow==2.9.1
RUN pip install -e ./taming-transformers
RUN pip install omegaconf>=2.0.0 pytorch-lightning>=1.0.8 torch-fidelity einops
RUN pip install transformers
RUN pip install dotmap
RUN pip install resize-right
RUN pip install piq
RUN pip install lpips
RUN pip install basicsr
RUN pip install facexlib
RUN pip install realesrgan
RUN pip install ipywidgets
RUN pip install opencv-python

RUN git clone https://github.com/apolinario/Multi-Modal-Comparators --branch gradient_checkpointing
RUN pip install poetry
WORKDIR /src/Multi-Modal-Comparators
RUN poetry build; pip install dist/mmc*.whl
WORKDIR /src
RUN python Multi-Modal-Comparators/src/mmc/napm_installs/__init__.py

RUN git lfs clone https://huggingface.co/datasets/multimodalart/latent-majesty-diffusion-settings

VOLUME [ "/src/outputs" ]
VOLUME [ "/root/.cache" ]

COPY *.py ./
COPY *.ipynb ./

ENTRYPOINT ["python", "latent.py", "--model_path", "/root/.cache/majesty-diffusion", "--model_source", "https://models.nmb.ai/majesty"]
