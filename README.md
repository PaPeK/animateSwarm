# animateSwarm – animation of active-agents

[Github](https://github.com/PaPeK/animateSwarm)

## General Notes


## Install

The package depends on a number of basic python packages. Either you can install the provided anaconda environment (see next subsection) or directly install in your current environment (which should automatically install the needed packages via pip).

### Required python packages

The code runs in the anaconda environment specified in `environment.yml` which can be created from the projects root directory via
```shell
conda env create -f environment.yml
```
which creates the environment `animateSwarm` (name specified in `environment.yml`) which you activate by
```shell
conda activate animateSwarm
```
In case you want to update your python packages to enable the code to run the most important packages are:

- numpy
- h5py
- scipy
- matplotlib
- pathlib

### Install in environment

Install locally via running 
```
pip install -e . --user
```
in the root-directory of this package (animateSwarm) where the `setup.py` file is located.
Now you can import the package in your jupyter-notebook or python script via

```python
from animateSwarm import AnimateTools as at
``` 

## Examples 

After you installed the package you can change to the subfolder `examples` and run

```
python AnimateExamples.py
```

which will animate the data from the directory `data` in 3 different animations.

Alternatively you start the ipython-notebook `examples/AnimateExamples.ipynb` and run the cells.


## User Agreement

By downloading SDE_burst_coast you agree with the following points: SDE_burst_coast is provided without any warranty or conditions of any kind. We assume no responsibility for errors or omissions in the results and interpretations following from application of SDE_burst_coast.

## License

Copyright (C) 2016-2020 Pascal Klamser

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
