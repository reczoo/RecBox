# MatchBox

Industrial recommender systems generally have two main stages: matching and ranking. In the first stage, candidate item matching (also known as candidate retrieval) aims for efficient and high-recall retrieval from a large item corpus. MatchBox provides an open source library for candidate item matching, with stunning features in configurability, tunability, and reproducibility. 


## Model Zoo

| Publication | Model          | Paper                                    | Benchmark | 
|:-----------:|:--------------:|:----------------------------------------------------------------- |:-------------:|
| UAI'09      | [MF-BPR](./model_zoo/MF)         | [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)                            | [:arrow_upper_right:](./model_zoo/MF/config) |
| RecSys'16   | [YoutubeNet](./model_zoo/YoutubeNet)        | [Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/doi/10.1145/2959100.2959190)                                            | [:arrow_upper_right:](./model_zoo/YouTubeNet/config) |
| CIKM'21     | [MF-CCL](./model_zoo/MF)/ [SimpleX](./model_zoo/SimpleX)  | [SimpleX: A Simple and Strong Baseline for Collaborative Filtering](https://arxiv.org/abs/2109.12613) | [:arrow_upper_right:](./model_zoo/SimpleX/config) | 


## Dependency

We suggest to use the following environment where we test MatchBox only. 

+ CUDA 10.0
+ python 3.6
+ pytorch 1.0
+ PyYAML
+ pandas
+ scikit-learn
+ numpy
+ h5py
+ tqdm


## Get Started

The code workflow is structured as follows:

```python
# Set the dataset config and model config
feature_cols = [{...}] # define feature columns
label_col = {...} # define label column
params = {...} # set data params and model params

# Set the feature encoding specs
feature_encoder = FeatureEncoder(feature_cols, label_col, ...) # define the feature encoder
datasets.build_dataset(feature_encoder, ...) # fit feature_encoder and build dataset 

# Load data generators
train_gen, valid_gen, test_gen = datasets.h5_generator(feature_encoder, ...)

# Define a model
model = SimpleX(...)

# Train the model
model.fit(train_gen, valid_gen, ...)

# Evaluation
model.evaluate(test_gen)
```

#### Run the code

For reproducing the experiment results, you can run the benchmarking script with the corresponding configs as follows.

+ --config: The config directory where dataset config and model config are located.
+ --expid: The experiment id defined in a model config file to set a group of hyper-parameters.
+ --gpu: The gpu index used for experiment, and -1 for CPU.

```bash
cd model_zoo/SimpleX
python run_expid.py --config ./config/SimpleX_yelp18_m1 --expid SimpleX_yelp18_m1 --gpu 0
python run_expid.py --config ./config/SimpleX_amazonbooks_m1 --expid SimpleX_amazonbooks_m1 --gpu 0
python run_expid.py --config ./config/SimpleX_gowalla_m1 --expid SimpleX_gowalla_m1 --gpu 0
```

The running logs are also available in each config directory.

