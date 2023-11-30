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
from tensorflow.keras.models import Model
from recbox.metrics import compute_rank_metrics
from recbox.utils.tf_utils import get_optimizer, get_loss
from recbox.utils import Monitor
import logging
from tqdm import tqdm


class RankingModel(Model):
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
        self._task = task
        self.feature_map = feature_map
        self.model_id = model_id
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + ".model"))
        self.validation_metrics = kwargs["metrics"]

    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, lr)
        self.loss_fn = get_loss(loss)

    def build_model(self, inputs, outputs):
        """ override to build the model with inputs and outputs
        """
        super(RankingModel, self).__init__(inputs=inputs, outputs=outputs)

    def compute_loss(self, return_dict, y_true):
        loss = self.loss_fn(return_dict["y_pred"], y_true)
        loss += sum(self.losses) # with regularization
        return loss
        
    def get_inputs(self, inputs={}, feature_source=[]):
        if type(feature_source) == str:
            feature_source = [feature_source]
        input_dict = dict()
        for feature, spec in self.feature_map.features.items():
            if feature_source and (spec["source"] not in feature_source):
                continue
            if spec["type"] == "meta":
                continue
            elif spec["type"] == "numeric":
                input_dict[feature] = inputs.get(feature, tf.keras.Input(shape=(1,), dtype=tf.float32))
            elif spec["type"] == "categorical":
                input_dict[feature] = inputs.get(feature, tf.keras.Input(shape=(1,), dtype=tf.int64))
            elif spec["type"] == "sequence":
                input_dict[feature] = inputs.get(feature, tf.keras.Input(shape=(spec["max_len"],), dtype=tf.int64))
        for label in self.feature_map.labels:
            input_dict[label] = inputs.get(label, tf.keras.Input(shape=(1,), dtype=tf.float32))
        return input_dict

    def get_labels(self, inputs):
        """ `Please override get_labels() when using multiple labels!`
        """
        labels = self.feature_map.labels
        y = inputs[labels[0]]
        return y

    def get_group_id(self, inputs):
        return inputs[self.feature_map.group_id]

    def lr_decay(self, factor=0.1, min_lr=1e-6):
        self.optimizer.learning_rate = max(self.optimizer.learning_rate * factor, min_lr)
        return self.optimizer.lr.numpy()
           
    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0

        # logging.info("Start training: {} batches/epoch".format(self._batches_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self.train_epoch(data_generator)
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)

    def train_epoch(self, data_generator):
        self._batch_index = 0
        train_loss = 0
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            loss = self.train_step(batch_data)
            train_loss += loss.numpy()
            if (self._eval_n_steps is not None) and (self._total_steps % self._eval_n_steps == 0):
                logging.info("Train loss: {:.6f}".format(train_loss / self._eval_n_steps))
                train_loss = 0
                self.eval_step()
            if self._stop_training:
                break
        if self._eval_n_steps is None:
            logging.info("Train loss: {:.6f}".format(train_loss / (self._batch_index + 1)))
            self.eval_step()

    @tf.function
    def train_step(self, batch_data):
        batch_data = self.get_inputs(inputs=batch_data)
        y_true = self.get_labels(batch_data)
        with tf.GradientTape() as tape:
            return_dict = self(batch_data, training=True)
            loss = self.compute_loss(return_dict, y_true)
            grads = tape.gradient(loss, self.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, self._max_gradient_norm)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def evaluate(self, data_generator, metrics=None):
        y_pred = []
        y_true = []
        group_id = []
        if self._verbose > 0:
            data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_data in data_generator:
            return_dict = self(batch_data, training=True)
            y_pred.extend(return_dict["y_pred"].numpy().reshape(-1))
            y_true.extend(self.get_labels(batch_data).numpy().reshape(-1))
            if self.feature_map.group_id is not None:
                group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
        y_pred = np.array(y_pred, np.float64)
        y_true = np.array(y_true, np.float64)
        group_id = np.array(group_id) if len(group_id) > 0 else None
        if metrics is not None:
            val_logs = self.compute_metrics(y_true, y_pred, metrics, group_id)
        else:
            val_logs = self.compute_metrics(y_true, y_pred, self.validation_metrics, group_id)
        logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
        return val_logs

    def eval_step(self):
        logging.info('Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())
        self.check_earlystop(val_logs)

    def check_earlystop(self, logs, min_delta=1e-6):
        monitor_value = self._monitor.get_value(logs)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
           (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({})={:.6f} STOP!".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                current_lr = self.lr_decay()
                logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info("Save best model: monitor({})={:.6f}"\
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps >= self._early_stop_patience:
            self._stop_training = True
            logging.info("********* Epoch=={} early stop *********".format(self._epoch_index + 1))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)

    def compute_metrics(self, y_true, y_pred, metrics, group_id=None):
        return compute_rank_metrics(y_true, y_pred, metrics, group_id)

    def output_activation(self, x):
        if self._task == "binary_classification":
            return tf.sigmoid(x)
        elif self._task == "regression":
            return x
        else:
            raise NotImplementedError("task={} is not supported.".format(self._task))
