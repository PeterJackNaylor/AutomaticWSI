# README

This repository contains the code necessary to reproduce the work presented in "[...]"

# Code structure

The code is based on nextflow which allows to submit process to a cluster scheduler in a automatic way. In particular, nextflow is smart enough to know when to relaunch jobs and when to cache them. In particular, it allows for a structured output. 

For each project, each nextflow file takes a project name and version and creates the correct output folder structures. 
Nextflow files call python files that can be found in the 'python' folder and each process has its own folder. In script, you will find the bash script used for launching the process. Each project should have all associated bash files and the sufficient lines to reproduce the project from this repository.

In test, you will find unit tests for some functions but also checks, intermediate controls and plotting function that allow to explore the intermediate analysis.

# Project Test
'''
home_tiling.sh
home_model_2S.sh
'''

# Project work outline on thalassa:

 1) Run 'bash script/thalassa_tiling.sh' to create the tile encoding folders.

 In particular we have in `outputs/{PROJECT_NAME}/tiling/{analysis_level}` the folders: `info`, `mat`, `mean`, `visu`.

 All tissue-tile encoding can be found in `mat`. 
 Tissue encoded mean can be found in `mean`.

## Done
- file 'Tiling-encoding.nf'

## In progress
- Model_2S.nf

## TODO
- Weldon/chowder done?
- Model 1S
- CONAN

:::

```python
import abc
```