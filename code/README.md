# Code repository

This repository contains the code for the paper *"Investigation of ensemble methods for the detection of deepfake face manipulations"* (https://arxiv.org/abs/2304.07395), which was created in the context of the DeepFakeChain project. The code compares different ensemble architectures for the tasks of deepfake detection and attribution, for both intra- and cross-domain evaluation scenaria. In particular, it considers the following architectures:

- **binary ensemble**: an ensemble of N binary detectors that detect fake from real faces, without distinguishing the manipulation type. The decisions are averaged across all detectors.
- **multiclass ensemble**: an ensemble of N multiclass detectors that distinguish the manipulation type of the fake faces. The decisions are averaged across all detectors.
- **one-vs-real ensemble**: an ensemble of N binary detectors that specialize in a particular manipulation against real faces. The decision with the highest score is selected as the ensemble's output.
- **one-vs-rest ensemble**: an ensemble of N binary detectors that specialize in a particular manipulation against all other manipulations and real faces. The decision with the highest score is selected as the ensemble's output.

Further details can be found in the paper.

The code has the following structure:

  **`datasets`**: contains the datasets used for training and evaluation. In our paper, we consider *FaceForensics++*  for training and testing, as well as *CelebDF*, *DFDCpreview*, *DFDC*, and *OpenForensics* for cross-domain evaluation. For the generation scripts to work, the datasets must be downloaded and the the frames must be sampled from the video sources (PENDING: include code for frame sampling).
  
**`inputs`**: contains code for handling different datasets. The `sources.py` generates the image paths and labels for each dataset, while the `datasets.py`  encapsulates all data in the ImagePathDataset (a PyTorch Dataset module), which allows handling all datasets uniformly.

 **`experiments`**: contains the definitions of the PyTorch model architectures. The `single_model` package defines the base models that are used in the ensembles, for either detection or attribution. The `ensemble_avgpool` package defines an ensemble that averages the outputs of its models, which is used in the binary and multiclass ensembles. The `ensemble_maxpool` package defines an ensemble that selects the manipulation with the highest score, which is used in the one-vs-real and one-vs-rest ensembles.

 **`models`**: contains the trained model in .pt form, along with hyperparameters.

 **`scripts`**: contains the scripts for generating the results of our paper.
 
**`outputs`**: stores the results of the scripts as text files.

