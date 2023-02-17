import torch.nn as nn
import numpy as np
import torch
import os
import sys
import logging
from tqdm import tqdm
from ...metrics import evaluate_metrics
from ..torch_utils import set_device, set_optimizer, set_loss, set_regularizer
from ...utils import Monitor
from .. import losses

class BaseModel(nn.Module):
    def __init__(self, 
                 feature_map, 
                 model_id="BaseModel", 
                 gpu=-1, 
                 monitor="AUC", 
                 save_best_only=True, 
                 monitor_mode="max", 
                 patience=2, 
                 eval_interval_epochs=1, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 reduce_lr_on_plateau=True, 
                 embedding_initializer="lambda w: nn.init.normal_(w, std=1e-4)", 
                 num_negs=0,
                 **kwargs):
        super(BaseModel, self).__init__()
        self.device = set_device(gpu)
        self.feature_map = feature_map
        self._monitor = Monitor(kv=monitor)
        self._monitor_mode = monitor_mode
        self._patience = patience
        self._eval_interval_epochs = eval_interval_epochs # float acceptable
        self._save_best_only = save_best_only
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self._embedding_initializer = embedding_initializer
        self.model_id = model_id
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + ".model"))
        self._validation_metrics = kwargs["metrics"]
        self._verbose = kwargs["verbose"]
        self.num_negs = num_negs

    def compile(self, lr=1e-3, optimizer=None, loss=None, **kwargs):
        try:
            self.optimizer = set_optimizer(optimizer)(self.parameters(), lr=lr)
        except:
            raise NotImplementedError("optimizer={} is not supported.".format(optimizer))
        if loss == "SigmoidCrossEntropyLoss":
            self.loss_fn = losses.SigmoidCrossEntropyLoss()
        elif loss == "PairwiseLogisticLoss":
            self.loss_fn = losses.PairwiseLogisticLoss()
        elif loss == "SoftmaxCrossEntropyLoss":
            self.loss_fn = losses.SoftmaxCrossEntropyLoss()
        elif loss == "PairwiseMarginLoss":
            self.loss_fn = losses.PairwiseMarginLoss(margin=kwargs.get("margin", 1))
        elif loss == "MSELoss":
            self.loss_fn = losses.MSELoss()
        elif loss == "CosineContrastiveLoss":
            self.loss_fn = losses.CosineContrastiveLoss(margin=kwargs.get("margin", 0),
                                                        negative_weight=kwargs.get("negative_weight"))
        else:
            raise NotImplementedError("loss={} is not supported.".format(loss))
        self.apply(self.init_weights)
        self.to(device=self.device)

    def get_total_loss(self, y_pred, y_true):
        # y_pred: N x (1 + num_negs) 
        # y_true:  N x (1 + num_negs) 
        y_true = y_true.float().to(self.device)
        total_loss = self.loss_fn(y_pred, y_true)
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = set_regularizer(self._embedding_regularizer)
            net_reg = set_regularizer(self._net_regularizer)
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if "embedding_layer" in name:
                        if self._embedding_regularizer:
                            for emb_p, emb_lambda in emb_reg:
                                total_loss += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
                    else:
                        if self._net_regularizer:
                            for net_p, net_lambda in net_reg:
                                total_loss += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return total_loss

    def init_weights(self, m):
        if type(m) == nn.ModuleDict:
            for k, v in m.items():
                if type(v) == nn.Embedding:
                    if "pretrained_emb" in self.feature_map.feature_specs[k]: # skip pretrained
                        continue
                    try:
                        initialize_emb = eval(self._embedding_initializer)
                        if v.padding_idx is not None:
                            # using the last index as padding_idx
                            initialize_emb(v.weight[0:-1, :])
                        else:
                            initialize_emb(v.weight)
                    except:
                        raise NotImplementedError("embedding_initializer={} is not supported."\
                                                  .format(self._embedding_initializer))
                elif type(v) == nn.Linear:
                    nn.init.xavier_normal_(v.weight)
                    if v.bias is not None:
                        v.bias.data.fill_(0)
        elif type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        
    def to_device(self, inputs):
        self.batch_size = 0
        for k in inputs.keys():
            inputs[k] = inputs[k].to(self.device)
            if self.batch_size < 1:
                self.batch_size = inputs[k].size(0)
        return inputs

    def on_batch_end(self, train_generator, batch_index, logs={}):
        self._total_batches += 1
        if (batch_index + 1) % self._eval_interval_batches == 0 or (batch_index + 1) % self._batches_per_epoch == 0:
            val_logs = self.evaluate(train_generator, self.valid_gen)
            epoch = round(float(self._total_batches) / self._batches_per_epoch, 2)
            self.checkpoint_and_earlystop(epoch, val_logs)
            logging.info("--- {}/{} batches finished ---".format(batch_index + 1, self._batches_per_epoch))

    def reduce_learning_rate(self, factor=0.1, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            reduced_lr = max(param_group["lr"] * factor, min_lr)
            param_group["lr"] = reduced_lr
        return reduced_lr

    def checkpoint_and_earlystop(self, epoch, logs, min_delta=1e-6):
        monitor_value = self._monitor.get_value(logs)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
           (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({}) STOP: {:.6f} !".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                current_lr = self.reduce_learning_rate()
                logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))
                logging.info("Load best model: {}".format(self.checkpoint))
                self.load_weights(self.checkpoint)
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info("Save best model: monitor({}): {:.6f}"\
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps * self._eval_interval_epochs >= self._patience:
            self._stop_training = True
            logging.info("Early stopping at epoch={:g}".format(epoch))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)
     
    def fit(self, train_generator, epochs=1, valid_generator=None,
            verbose=0, max_gradient_norm=10., **kwargs):
        self.valid_gen = valid_generator
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._total_batches = 0
        self._batches_per_epoch = len(train_generator)
        self._eval_interval_batches = int(np.ceil(self._eval_interval_epochs * self._batches_per_epoch))
        self._stop_training = False
        self._verbose = verbose
        
        logging.info("**** Start training: {} batches/epoch ****".format(self._batches_per_epoch))
        for epoch in range(epochs):
            epoch_loss = self.train_on_epoch(train_generator, epoch)
            logging.info("Train loss: {:.6f}".format(epoch_loss))
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(epoch + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)

    def train_on_epoch(self, train_generator, epoch):
        epoch_loss = 0
        model = self.train()
        batch_generator = train_generator
        if self._verbose > 0:
            batch_generator = tqdm(train_generator, disable=False)#, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_generator):
            self.optimizer.zero_grad()
            return_dict = model.forward(batch_data)
            loss = return_dict["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            epoch_loss += loss.item()
            self.on_batch_end(train_generator, batch_index)
            if self._stop_training:
                break
        return epoch_loss / self._batches_per_epoch

    def evaluate(self, train_generator, valid_generator):
        logging.info("--- Start evaluation ---")
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            user_vecs = []
            item_vecs = []
            for user_batch in valid_generator.user_loader:
                user_vec = self.user_tower(user_batch)
                user_vecs.extend(user_vec.data.cpu().numpy())
            for item_batch in valid_generator.item_loader:
                item_vec = self.item_tower(item_batch)
                item_vecs.extend(item_vec.data.cpu().numpy())
            user_vecs = np.array(user_vecs, np.float64)
            item_vecs = np.array(item_vecs, np.float64)
            val_logs = evaluate_metrics(user_vecs,
                                        item_vecs,
                                        train_generator.user2items_dict,
                                        valid_generator.user2items_dict,
                                        valid_generator.query_indexes,
                                        self._validation_metrics)
            return val_logs
                
    def save_weights(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def load_weights(self, checkpoint):
        self.load_state_dict(torch.load(checkpoint, map_location=self.device))

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters(): 
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logging.info("Total number of parameters: {}.".format(total_params))

