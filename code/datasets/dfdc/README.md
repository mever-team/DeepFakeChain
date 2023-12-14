**DFDC** is a large-scale dataset for deepfake detection offered by Facebook (now Meta). It contains 124k videos and 8 manipulation methods. It is provided along a smaller preview version. Additional information about the DFDC and its preview version long with download instructions can be found in <https://ai.meta.com/datasets/dfdc/>.

### Instructions for preprocessing

- Download the dataset
- Extract all zipped files
- Run the [preprocess script](https://github.com/mever-team/DeepFakeChain/blob/main/code/scripts/preprocess/preprocess_dfdc.sh) from the [code](https://github.com/mever-team/DeepFakeChain/tree/main/code) directory passing the dataset's path as argument. The code will preprocess both DFDC and its preview version.
