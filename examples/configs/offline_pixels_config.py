import ml_collections
from ml_collections.config_dict import config_dict


def get_bc_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.cnn_features = (32, 64, 128, 256)
    config.cnn_filters = (3, 3, 3, 3)
    config.cnn_strides = (2, 2, 2, 2)
    config.cnn_padding = "VALID"
    config.latent_dim = 50

    config.encoder = "d4pg"

    config.dropout_rate = config_dict.placeholder(float)
    config.cosine_decay = True

    return config


def get_iql_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.value_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.cnn_features = (32, 64, 128, 256)
    config.cnn_filters = (3, 3, 3, 3)
    config.cnn_strides = (2, 2, 2, 2)
    config.cnn_padding = "VALID"
    config.latent_dim = 50

    config.discount = 0.99

    config.expectile = 0.7  # The actual tau for expectiles.
    config.A_scaling = 3.0
    config.dropout_rate = config_dict.placeholder(float)
    config.cosine_decay = True

    config.tau = 0.005

    config.critic_reduction = "min"
    config.share_encoder = False

    return config


def get_config(config_string):
    possible_structures = {
        "bc": ml_collections.ConfigDict(
            {"model_constructor": "PixelBCLearner", "model_config": get_bc_config()}
        ),
        "iql": ml_collections.ConfigDict(
            {"model_constructor": "PixelIQLLearner", "model_config": get_iql_config()}
        ),
    }
    return possible_structures[config_string]
