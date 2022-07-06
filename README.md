# TwinX
TwinX is an open source library for two-tower matching with stunning features in configurability, tunability, and reproducibility. 

## Model List

| Publication |    Model   |  Paper                                                                                       |
| :----:|:----------:|:--------------------------------------------------------------------------------------------|
| UAI'09 |   MF-BPR   |      [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)                         |
| RecSys'16 | YoutubeDNN |    [Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/doi/10.1145/2959100.2959190)                               |
| CIKM'21 |    MF-CCL / SimpleX    |    SimpleX: A Simple and Strong Baseline for Collaborative Filtering  |


## Get Started

#### Run the demo

The code workflow is structured as follows:

```python
# Set the data config and model config
feature_cols = [{...}] # define feature columns
label_col = {...} # define label column
params = {...} # set data params and model params

# Set the feature encoding specs
feature_encoder = FeatureEncoder(feature_cols, label_col, ...) # define the feature encoder
datasets.build_dataset(feature_encoder, ...) # fit feature_encoder and build dataset 

# Load data generators
train_gen, valid_gen, test_gen = h5_generator(feature_encoder, ...)

# Define a model
model = SimpleX(...)

# Train the model
model.fit(train_gen, valid_gen, ...)

# Evaluation
model.evaluate(test_gen)

```

#### Run the benchmark

For reproducing the experiment result, you can run the benchmarking script with the corresponding config file as follows.

+ --config: The config file that defines the hyper-parameter space.
+ --gpu: The gpu index used for experiment, and -1 for CPU.

```bash
cd benchmarks
python run_param_tuner.py --config Yelp18/SimpleX_yelp18_x0/SimpleX_yelp18_x0_tuner_config.yaml --gpu 0

```




