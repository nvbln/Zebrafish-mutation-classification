# Zebrafish Mutation Classification
Part of an internship at Portugues' lab. The goal of the internship is to build a RNN to classify zebrafish as either healthy or mutated based on neuronal recordings.

## Installation
The installation exists of two parts. One part is setting up the dataset,
the other part is setting up the python virtual environment.

The expected format of the dataset is a series of directories, each
containing a different subject, stored in the `./data` directory. The
directory must contain a `*_behavior_log.hdf5` and `*_metadata.json` file.
If these files are missing, the directory will be skipped. If the directory
contains multiple files, the biggest behavior file in combination with its
metadata file will be selected. The data is expected to come from Stytra
and thus formatted in the way that Stytra saves its data files. Any
recordings from Stytra (containing behavioural recordings) can be put in
the data directory and used as input to the model with very little effort.

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

Note that Pytorch is among the requirements and should be automatically
installed, but might not always work as expected. If any issues are
encountered, please uninstall it and then reinstall it in the following
way:

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
