# Installation

## Install CARLA
- Download and unzip [CARLA 0.9.10.1](https://github.com/carla-simulator/carla/releases/tag/0.9.10.1)

## Environment setup

- Clone this repo and setup the conda environment, our code was developed with PyTorch 1.12 and CUDA 11.3.
```
git clone https://github.com/h2xlab/CaT
cd CaT

conda create -n cat-env python=3.8 -y

# install pytorch [CUDA 11.3]
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# install torch-scatter
pip install torch-scatter==2.0.7 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

# install remaining dependencies
pip install -r requirements.txt


cd ./cat/models/nms
python setup.py install

cd ../ops/
bash make.sh

```

## Configure environment variables

Configure environment variables by running the following inside this repo
```
#!/bin/bash
export CARLA_ROOT=[Path to CARLA]
export LEADERBOARD_ROOT=leaderboard
export SCENARIO_RUNNER_ROOT=scenario_runner
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:"${CARLA_ROOT}"/PythonAPI/carla:"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}"

export SCENARIOS="${LEADERBOARD_ROOT}"/data/all_towns_traffic_scenarios_public.json
export TEAM_AGENT=team_code/cat_agent.py
export TEAM_CONFIG=config.yaml

export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export CHECKPOINT_ENDPOINT=results_submit.json
```
