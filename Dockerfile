FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /usr/src/app

COPY . /usr/src/app/

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y supervisor autoconf automake autopoint libtool libhunspell-dev
                                                    
RUN git clone https://github.com/MSeal/cython_hunspell.git
RUN python3 -m pip install cython_hunspell/
RUN python3 -m pip install --no-cache-dir -r ./requirements.txt
RUN python3 -m pip install ctranslate2==4.4.0

COPY ./supervisord.conf /etc/supervisor/conf.d/supervisord.conf
RUN mkdir -p /var/run/supervisor
# Defina o LD_LIBRARY_PATH dinamicamente
RUN export LIB_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))') && \
    sed -i "/\[program:app1\]/a environment=LD_LIBRARY_PATH=\"$LIB_PATH:\$LD_LIBRARY_PATH\"" /etc/supervisor/conf.d/supervisord.conf

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
