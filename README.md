# Decagon implementation used in "Predicting Adverse Drug-Drug Interactions with Neural Embedding of Semantic Predications"

#### Repository author: Hannah Burkhardt (haalbu@uw.edu)

This repository contains the implementation of the [Decagon algorithm](https://doi.org/10.1093/bioinformatics/bty294) used in our paper:

Burkhardt, Hannah A, Devika Subramanian, Justin Mower, and Trevor Cohen. 2019. “Predicting Adverse Drug-Drug Interactions with Neural Embedding of Semantic Predications.” To Appear in Proc AMIA Annu Symp 2019.

The required Decagon data files are available at [http://snap.stanford.edu/decagon](http://snap.stanford.edu/decagon).

Please also see the companion repository, [Predicting DDIs with ESP](https://github.com/hannahburkhardt/predicting_ddis_with_esp), to download the code for the ESP procedure.
  
## Usage

```bash
git clone https://github.com/hannahburkhardt/decagon.git
cd decagon
```

After cloning the repository, create a new conda environment with the given configuration like so:
```bash
conda create -n decagon_ddi --file decagon_env_spec_file.txt python=3.6.8
``` 

If you haven't already, download the bio-decagon data files from the Decagon project website, e.g. into `bio-decagon`:

```bash
mkdir bio-decagon
cd bio-decagon
wget http://snap.stanford.edu/decagon/bio-decagon-ppi.tar.gz http://snap.stanford.edu/decagon/bio-decagon-combo.tar.gz http://snap.stanford.edu/decagon/bio-decagon-mono.tar.gz http://snap.stanford.edu/decagon/bio-decagon-targets-all.tar.gz
for i in *.tar.gz; do tar -zxvf $i; done
cd ..

```
Next, run Decagon, e.g. like so:
```bash
conda activate decagon_ddi
python main.py --decagon_data_file_directory ../bio-decagon/ --epochs=4
```
Depending on hardware, the running time will be approximately 7 hours + epochs*36 hours, that is about 6 days for 4 epochs.
