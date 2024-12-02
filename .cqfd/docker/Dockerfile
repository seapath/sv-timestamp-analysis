FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
  locales \
  python3-pandas \
  python3-numpy \
  python3-matplotlib

RUN sed -i "s/# en_US\.UTF-8 UTF-8/en_US\.UTF-8 UTF-8/" /etc/locale.gen
RUN locale-gen
RUN dpkg-reconfigure locales
