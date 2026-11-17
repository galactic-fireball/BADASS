#!/usr/bin/env bash

git pull origin
python3 -m pip uninstall -y spark
python3 -m pip install -e .
