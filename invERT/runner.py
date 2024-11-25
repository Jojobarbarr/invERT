from config.configuration import Config
from model.models import DynamicModel







def main(config: Config):
    # Data
    dataset_config: Config = config.dataset
    if dataset_config.dataset_name == "_generated":
        min_shape: int = dataset_config.min_shape




    mlp_config: Config = config.model.mlp
    cnn_config: Config = config.model.cnn


    num_filters: list[int] = []
    kernel_sizes: list[int] = []
    for index, conv_layer in enumerate(cnn_config.conv_layers):
        num_filters.append(conv_layer[index].filters) # /!\ last filter number must be 1
        kernel_sizes.append(conv_layer[index].kernel_size) 
    model = DynamicModel(mlp_config.input_size, mlp_config.hidden_layers, num_filters, kernel_sizes, 1)

if __name__ == "__main__":
    pass