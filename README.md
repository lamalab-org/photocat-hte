# Simple tooling for running a photocatalysis HTE

## How to 

0. Set up experimental config. 
    a. `experiments.yml` defines the experiments 
    b. `setup.yml` defines the setup (firesting, power supply, folders)
1. Set up experimental setup 
2. Start AutoSuite Program, ensure that there is no `values_for_experiment.csv` in the folder that is left over. The AutoSuite program will read that and start from there, even if the Python code is not running. 
3. In the `base` Python environment run `python run.py` in the `experiments` folder of this repo


## Config files 

### `experiments.yml` 
This file lists experimental conditions. If `run: false`, the condition will not be run (this is read by the Chemspeed AutoSuite and then leaves the code in a `while` loop where nothing physically happens). 

The meaning of the parameters is:

- `name`: human-readable name of the experiments. Will be used for naming the log file 
- `run` : determiner if the experiment defined by the values below is run (run : true) or not (run : false) --> utilized to set a break in the code to make it possible to change the lid of the vial after three performed reactions
- `voltage`: voltage in `V` for the light source (ensure that a minimum current of 0.3 A is set at the programmable power source(do this manually at the power source))
- `volume_water`: the volume of water in `mL`
- `volume_buffer__base`: the volume of the basic component of the buffer  in `mL`
- `volume_buffer_acid_`: the volume of the acidic component of the buffer  in `mL`
- `volume_sacrificial_oxidant`: the volume of sacrificial oxidant in `mL` 
- `volume_photosensitizer`: the volume of the solution containing the photosensitizer in `mL`
- `volume_catalyst`: the volume of the solution containing the catalyst in `mL`
- `degassing_time`: time for degassing of the reaction solution prior to irradiation in `min`
- `measurement_time`: time for measurement of the reaction while irradiated in `min`
- `pre_reaction_baseline_time`: time in minutes for waiting prior to reaction
- `post_reaction_baseline_time`: time in minutes for waiting post reaction


An example file looks like 

```yaml
- 
  name: MRG-059-T
  run: "true"
  voltage: 0.18
  volume_water: 5.1
  volume_buffer_base: 0.85
  volume_buffer_acid: 0.85
  volume_sacrificial_oxidant: 0.85
  volume_photosensitizer:  0.85
  volume_catalyst:  0.85
  degassing_time: 20
  measurement_time: 10 
  pre_reaction_baseline_time: 15
  post_reaction_baseline_time: 5
- 
  name: MRG-059-T
  run: "true" 
  voltage: 0.18
  volume_water: 5.1
  volume_buffer_base: 0.85
  volume_buffer_acid: 0.85
  volume_sacrificial_oxidant: 0.85
  volume_photosensitizer:  0.85
  volume_catalyst:  0.85
  degassing_time: 20
  measurement_time: 10 
  pre_reaction_baseline_time: 15
  post_reaction_baseline_time: 5
```


### `setup.yml`

The `setup.yml` defines global settings such as directories for logging as well as ports for lamp and sensor.