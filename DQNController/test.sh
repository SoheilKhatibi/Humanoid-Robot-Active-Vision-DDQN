#!/bin/bash

export WEBOTS_HOME="/usr/local/webots"
export LD_LIBRARY_PATH=":${WEBOTS_HOME}/lib/controller"
export PYTHONPATH=":${WEBOTS_HOME}/lib/controller/python36"
export PYTHONIOENCODING="UTF-8"

python3.6 DQNController.py
