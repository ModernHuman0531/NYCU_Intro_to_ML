# HW3.Regression
## Prerequisite
* Have nvidia GPU on your device
* Install `python 3.13.7`
## Usage
Create virtual python and download python packages.
```zsh
make
```
Enable virtual environment
```zsh
source venv/bin/activate
```
If you want to add more python package,change content in `requirements.txt`, and run
```zsh
make install-requirements
```
And choose `My project Kernal` at the up right space, if you don't see that, manually run 
```zsh
make kernal
```
