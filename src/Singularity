Bootstrap: docker

From: continuumio/miniconda3

%files
    environment.yml

%post
    /usr/bin/apt-get update && /usr/bin/apt-get -y upgrade
    /usr/bin/apt-get update --fix-missing
    /usr/bin/apt-get install -y --no-install-recommends apt-utils
    /usr/bin/apt-get install -yq texlive-full
    /opt/conda/bin/conda env create -f environment.yml

%runscript
    exec /opt/conda/envs/river/bin/"$@"
