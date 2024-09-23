# ProSynth

### Setup ###
* Download the ProSynth source code 

* Create virtual environment through anaconda
```
conda create --name ProSynthEnv python=3.8
conda activate ProSynthEnv
```
* Install packages
   
```
cd ProSynth
pip install -r requirements.txt
```

* Install [Timeloop](https://timeloop.csail.mit.edu/timeloop)

### Run ProSynth on NVDLA ###

```
cd Accelerators/NVDLA
sh run.sh
sh run_ea.sh
```
