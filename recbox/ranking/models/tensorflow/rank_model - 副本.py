# =========================================================================
# Copyright (C) 2023. The RecBox Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import os
import sys
import numpy as np
import tensorflow as tf

import logging


class RankingModel(object):
    def __init__(self,
                 feature_map,
                 model_id="RankingModel",
                 task="binary_classification",
                 monitor="AUC",
                 save_best_only=True,
                 monitor_mode="max",
                 early_stop_patience=2,
                 eval_n_steps=None,
                 reduce_lr_on_plateau=True,
                 **kwargs):
        super(RankingModel, self).__init__()
        self.valid_gen = None
        self._monitor_mode = monitor_mode
        self._monitor = Monitor(kv=monitor)
        self._early_stop_patience = early_stop_patience
        self._eval_n_steps = eval_n_steps # None default, that is evaluating every epoch
        self._save_best_only = save_best_only
        self._verbose = kwargs["verbose"]
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self.feature_map = feature_map
        self.output_activation = self.get_output_activation(task)
        self.model_id = model_id
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + ".model"))
        self.validation_metrics = kwargs["metrics"]

    def build(self):
        """ override to build the model with inputs and outputs
            inputs = ...
            outputs = ...
            return Model(inputs=inputs, outputs=outputs)
        """
        pass

    def get_inputs(self, inputs, feature_source=None):
        if feature_source and type(feature_source) == str:
            feature_source = [feature_source]
        X_dict = dict()
        for feature, spec in self.feature_map.features.items():
            if (feature_source is not None) and (spec["source"] not in feature_source):
                continue
            if spec["type"] == "meta":
                continue
            X_dict[feature] = inputs[feature]
        return X_dict

    def get_labels(self, inputs):
        labels = self.feature_map.labels
        assert len(labels) == 1, "Please override get_labels(), add_loss(), evaluate() when using multiple labels!"
        y = inputs[labels[0]]
        return y

    def get_output_activation(self, task):
        if task == "binary_classification":
            return tf.keras.layers.Activation("sigmoid")
        elif task == "regression":
            return tf.identity
        else:
            raise NotImplementedError("task={} is not supported.".format(task))


