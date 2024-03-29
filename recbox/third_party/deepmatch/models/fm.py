"""
Author:
    Weichen Shen, weichenswc@163.com

"""
from deepctr.feature_column import build_input_features
from deepctr.layers.core import PredictionLayer
from deepctr.layers.utils import concat_func, reduce_sum
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.models import Model

from ..inputs import create_embedding_matrix, input_from_feature_columns
from ..layers.core import InBatchSoftmaxLayer
from ..utils import l2_normalize, inner_product


def FM(user_feature_columns, item_feature_columns, l2_reg_embedding=1e-6, loss_type='softmax', temperature=0.05,
       sampler_config=None, seed=1024,
       ):
    """Instantiates the FM architecture.

    :param user_feature_columns: An iterable containing user's features used by  the model.
    :param item_feature_columns: An iterable containing item's features used by  the model.
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param loss_type: string. Loss type.
    :param temperature: float. Scaling factor.
    :param sampler_config: negative sample config.
    :param seed: integer ,to use as random seed.
    :return: A Keras model instance.

    """

    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    seed=seed,
                                                    seq_mask_zero=True)

    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, _ = input_from_feature_columns(user_features,
                                                               user_feature_columns,
                                                               l2_reg_embedding, seed=seed,
                                                               support_dense=False,
                                                               embedding_matrix_dict=embedding_matrix_dict)

    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    item_sparse_embedding_list, _ = input_from_feature_columns(item_features,
                                                               item_feature_columns,
                                                               l2_reg_embedding, seed=seed,
                                                               support_dense=False,
                                                               embedding_matrix_dict=embedding_matrix_dict)

    user_dnn_input = concat_func(user_sparse_embedding_list, axis=1)
    user_vector_sum = Lambda(lambda x: reduce_sum(x, axis=1, keep_dims=False))(user_dnn_input)
    user_vector_sum = l2_normalize(user_vector_sum)

    item_dnn_input = concat_func(item_sparse_embedding_list, axis=1)
    item_vector_sum = Lambda(lambda x: reduce_sum(x, axis=1, keep_dims=False))(item_dnn_input)
    item_vector_sum = l2_normalize(item_vector_sum)

    if loss_type == "logistic":
        score = inner_product(user_vector_sum, item_vector_sum, temperature)
        output = PredictionLayer("binary", False)(score)

    elif loss_type == "softmax":
        output = InBatchSoftmaxLayer(sampler_config._asdict(), temperature)(
            [user_vector_sum, item_vector_sum, item_features[sampler_config.item_name]])
    else:
        raise ValueError(' `loss_type` must be `logistic` or `softmax` ')

    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("user_embedding", user_vector_sum)

    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("item_embedding", item_vector_sum)

    return model
