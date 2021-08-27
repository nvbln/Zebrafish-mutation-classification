# Zebrafish Mutation Classification
Part of an internship at Portugues' lab. The goal of the internship is to build a RNN to classify zebrafish as either healthy or mutated based on neuronal recordings.

## Installation
The installation exists of two parts. One part is setting up the dataset,
the other part is setting up the python virtual environment.

_Still need to add how to setup the dataset._

The second part is a bit less straightforward to set up. As the `torch`
and `torchvision` packages have to be installed separately. See the end
of this section for instructions on how to do this.


Instructions for `virtualenv` are provided below. It should also be possible
to use `anaconda` but this has not been tested.

Create a python virtual environment and install the packages listed in
`requirements.txt`. Note that these requirements will be updated in the
future, so if something does not work, please check whether your virtual
environment is up to date. To create a python virtual environment and
install the required packages execute the following commands from the
main directory of this repository:
```
python3 -m virtualenv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

While setting this up, I used Python 3.7.3. You can find your version by
executing: `python3 --version` (or just `python --version` depending on
your OS). Generally, any version >= 3.5 should work.

If you do not have the virtual environment package installed, this can be
done as follows for any Debian derivative (i.e. Ubuntu, Mint):
```
apt install python3-virtualenv
```
or via pip:
```
pip3 install virtualenv
```

If you want to run the network on a GPU, you should run the following:
```
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

If you only want/can run the network on a CPU, installing the following
should suffice:
```
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

Then install Pytorch Lightning:
```
pip install pytorch-lightning
```

## Running the project

The whole project can be run from the Jupyter notebook: 
`MutationClassification.ipynb`. To do so, execute:
```
source .venv/bin/activate
jupyter notebook
```
and navigate to the notebook.

## Committing the Jupyter Notebook

Jupyter Notebook has a lot of metadata and output that it saves in the
file. This will contaminate the versioning in Git. To keep the `diff`'s as
clean as possible, remove the empty cells and run Cell > All Output > Clear.
