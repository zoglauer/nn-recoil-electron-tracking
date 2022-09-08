#!/bin/bash

# Setup nothing on savio
if [[ ${HOSTNAME} == *.brc ]]; then
  exit
fi

PENV=python-env

if [[ ${HOSTNAME} == thebe ]]; then
  export TMPDIR=/volumes/selene/tmp
fi

if [ -d ${PENV} ]; then
  rm -r ./${PENV}
fi
python3 -m venv ${PENV}
if [ "$?" != "0" ]; then exit 1; fi
. ${PENV}/bin/activate
if [ "$?" != "0" ]; then exit 1; fi

# Upgrade pip
python3 -m pip install --upgrade pip
if [ "$?" != "0" ]; then exit 1; fi

# Install tensorflow & torch the special way
if [ "$(uname -s)" == "Darwin" ]; then 
  # HDF5 is troublesome, thus do this first
  P=$(which port); P=${P%/bin/port}
  if [[ -f ${P}/lib/libhdf5.dylib ]]; then
    export HDF5_DIR=/opt/local/
    pip3 install h5py 
    if [ "$?" != "0" ]; then exit 1; fi 
  else
    P=$(which brew)
    if [[ -f ${P} ]]; then
      export HDF5_DIR=$(brew --prefix hdf5)
      pip install h5py
      if [ "$?" != "0" ]; then exit 1; fi
    else
      echo "ERROR: hdf5 must be installed either via macports or brew"
      exit 1
    fi
  fi

  pip3 install tensorflow-macos
  if [ "$?" != "0" ]; then exit 1; fi
else
  if [[ ${HOSTNAME} == thebe ]] || [[ ${HOSTNAME} == despina ]]; then
    pip3 install tensorflow-gpu==2.3
    if [ "$?" != "0" ]; then exit 1; fi
    pip3 install torch
    if [ "$?" != "0" ]; then exit 1; fi
  else 
    pip3 install tensorflow
    if [ "$?" != "0" ]; then exit 1; fi
    pip3 install torch 
    if [ "$?" != "0" ]; then exit 1; fi
  fi
fi

# All the default installations
pip3 install -r Requirements.txt
if [ "$?" != "0" ]; then exit 1; fi

