FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

# Module
WORKDIR /tmp
ADD ./setup.py /tmp
ADD ./selective_gp /tmp/selective_gp
ADD ./requirements.txt /tmp
RUN pip install -r requirements.txt
RUN pip install -e .

# Scripts
ADD ./demonstrations /tmp/demonstrations
ADD ./experiments /tmp/experiments
ADD ./notebooks /tmp/notebooks

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run jupyter notebook server
CMD ["jupyter", "notebook", "notebooks", "--port=8888", "--no-browser",\
"--ip=0.0.0.0", "--allow-root"]
