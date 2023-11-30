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
from tensorflow.python.keras.callbacks import Callback
from recbox.metrics import eval_rank_metrics
from recbox.utils.tf_utils import get_optimizer, get_loss
from recbox.utils.common import Monitor
import logging
from tqdm import tqdm


class BaseModel(Model):
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
        super(BaseModel, self).__init__()
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
        self.callbacks = [TrainingCallback(**kwargs)]

    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, lr)
        self.loss_fn = get_loss(loss)

    def add_loss(self, inputs):
        return_dict = self(inputs, training=True)
        y_true = self.get_labels(inputs)
        loss = self.loss(return_dict["y_pred"], y_true)
        return loss
    
    def get_total_loss(self, inputs):
        total_loss = self.add_loss(inputs) + sum(self.losses) # with regularization
        return total_loss

    def get_labels(self, inputs):
        labels = self.feature_map.labels
        y = inputs[labels[0]]
        return y

    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        steps_per_epoch = len(generator)
        super(BaseModel, self).fit(generator=generator, steps_per_epoch=steps_per_epoch, 
                                   epochs=epochs, verbose=verbose, callbacks=self.callbacks, 
                                   workers=workers, use_multiprocessing=use_multiprocessing, 
                                   shuffle=False, max_queue_size=max_queue_size)

    @tf.function
    def train_step(self, batch_data):
        with tf.GradientTape() as tape:
            loss = self.get_total_loss(batch_data)
            grads = tape.gradient(loss, self.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, self._max_gradient_norm)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def evaluate(self, data_generator, max_queue_size=10, workers=2,
                 use_multiprocessing=False, verbose=0):
        steps = len(data_generator)
        y_true = data_generator.get_labels()
        y_pred = self.predict(data_generator, steps=steps, max_queue_size=max_queue_size,
                                workers=workers, use_multiprocessing=use_multiprocessing,
                                verbose=verbose)
        result = eval_rank_metrics(y_true, y_pred, self.validation_metrics, group_id=None)
        return result


class TrainingCallback(Callback):
    def __init__(self, monitor='AUC', save_best_only=True, monitor_mode='max', patience=2, 
                 every_x_epoches=1, workers=2, **kwargs):
        self._every_x_epoches = every_x_epoches # float acceptable
        self._monitor = Monitor(kv=monitor)
        self._save_best_only = save_best_only
        self._mode = monitor_mode
        self._patience = patience
        self._best_metric = np.Inf if self._mode == 'min' else -np.Inf
        self._stopping_steps = 0
        self._total_batches = 0
        self._workers = workers
        
    def on_train_begin(self, logs={}):
        self._batches = self.params['steps']
        self._every_x_batches = int(np.ceil(self._every_x_epoches * self._batches))
        logging.info("**** Start training: {} batches/epoch ****".format(self._batches))

    def on_batch_end(self, batch, logs={}):
        self._total_batches += 1
        if (batch + 1) % self._every_x_batches == 0 or (batch + 1) % self._batches == 0:
            val_logs = self.model.evaluate_generator(self.model.valid_gen, self._workers)
            logs.update(val_logs)
            epoch = round(float(self._total_batches) / self._batches, 2)
            self.checkpoint_and_earlystop(epoch, val_logs)
            logging.info('******* {}/{} batches finished *******'.format(batch + 1, self._batches))

    def on_epoch_end(self, epoch, logs={}):
        if 'loss' in logs:
            logging.info('[Train] loss: {:.6f}'.format(logs['loss']))
        logging.info('************ Epoch={} end ************'.format(epoch + 1))

    def checkpoint_and_earlystop(self, epoch, logs):
        monitor_value = self._monitor.get_value(logs)
        if (self._mode == 'min' and monitor_value > self._best_metric) or \
           (self._mode == 'max' and monitor_value < self._best_metric):
            self._stopping_steps += 1
            logging.info('Monitor({}) drops: {:.6f}'.format(self._mode, monitor_value))
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info('Save best model: monitor({}): {:.6f}'.format(self._mode, monitor_value))
                self.model.save(self.model.checkpoint, overwrite=True)
        if self._stopping_steps * self._every_x_epoches >= self._patience:
            self.model.stop_training = True
            logging.info('Early stopping at epoch={:g}'.format(epoch))
        if not self._save_best_only:
            self.model.save(self.model.checkpoint, overwrite=True)


