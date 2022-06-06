# MLMI Practical Course: OCT Image Generation via Diffusion Models

Repository for OCT Project MLMI Practical Course from Summer Semester 2022 at TUM.

Current code is an extension of [guided-diffusion](https://github.com/openai/guided-diffusion).

## Scope of the project

The project aims to build a neural network to extract meaningful features from OCT Imaging.

The goal is to further use this trained model in other applications, such as automatic report generation based on images alone.

Our effort relies upon implementing Diffusion Models to generate new OCT images, with the expectation that the extracted features of the neural network are good enough to make meaning of new unseen data. If this is accomplished, other downstream tasks such as the automatic report generation might be feasible with use of this neural network.  

## How to run

### Requirements

* `python >= 3.8`
* Python environment manager, such as `pipenv`
* `pip >= 22.0.4`
* `Python MPI`: Instructions [here](https://nyu-cds.github.io/python-mpi/setup/)

### Step by step

1. Clone repo locally

```batch
git clone https://github.com/murilobellatini/mlmi-oct-diffusion.git
```

2. Move to local repo root

```batch
cd ./mlmi-oct-diffusion
```

3. Initiate python environment (example for `pipenv` below)

```batch
pipenv shell
```

4. Install dependencies

```batch
pip install -r requirements.txt
```

5. Either run notebooks or cli commands below.
6. Optional: For notebooks it might be required to install kernel profile. If so, it can be done with code below.

```batch
jupyter kernelspec install-self 
```


