# Machine-learning based recoilelectron tracking


## Setup Option 1: You only use the provided data in the data directory (default)

As of June 2022, Python 3.9 and 3.10 do not yet work with tensorflow on M1 macs

### Get the code

To get the code, just clone the git repository:
```
git clone https://github.com/zoglauer/nn-recoil-electron-tracking RecoilElectronTracking
```

### Creating the environment

One of the required packages is at the moment only available via pip, not any other python package manager. In addition we need specific versions of some popular packages such as numpy. Therefore, we will have to setup a virtualenv environment to run our specific python version. In the COSIMachineLearning directory, do:

```
bash setup.sh
```

This will create a python environment in python-env, activate it, and install all required packages.


### Using it

Remember to activate your python environment whenever you are switching to a new bash shell:
```
. python-env/bin/activate
```

Now you are ready.



## Setup Option 2: Full install to be able to create your won data

### Prerequisites:

* Make sure you have all development tools installed on your computer, such as python3, git, cmake, etc.
* On macOS only the python version downloaded from python.org will work, or the ones coming with brew or macports
* April 2022: Python 3.9 and 3.10 do not yet work with tensorflow on M1 macs


### Install MEGAlib -- but only if it is required for your specific tool:

Follow [these instructions](http://megalibtoolkit.com/setup.html), to install MEGAlib.


### Get the code

To get the code, just clone the git repository:
```
git clone https://github.com/zoglauer/nn-recoil-electron-tracking RecoilElectronTracking
```


### Creating the environment

One of the required packages is at the moment only available via pip, not any other python package manager. In addition we need specific versions of some popular packages such as numpy. Therefore, we will have to setup a virtualenv environment to run our specific python version. In the COSIMachineLearning directory, do:

```
bash setupi-full.sh
```

This will create a python environment in python-env, activate it, and install all required packages.


### Using it

Remember to activate your python environment whenever you are switching to a new bash shell:
```
. python-env/bin/activate
```

Now you are ready. 





