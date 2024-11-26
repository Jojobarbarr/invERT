from torch.optim import Adam, SGD, RMSprop
from torch.nn import MSELoss, L1Loss

from config.configuration import Config
from model.models import DynamicModel
from data.data import generate_data, pre_process_data, initialize_datasets







def main(config: Config):
    # Data
    dataset_config: Config = config.dataset
    if dataset_config.dataset_name == "_generated":
        min_shape: int = dataset_config.data_min_size
        max_shape: int = dataset_config.data_max_size
        num_samples: int = dataset_config.num_samples
        noise: float = dataset_config.noise

        data = generate_data(num_samples, min_shape, max_shape, noise)

    data, target, min_data, max_data, min_target, max_target = pre_process_data(data)

    train_dataloader, test_dataloader, val_dataloader = initialize_datasets(data, target, dataset_config.batch_size, dataset_config.test_split, dataset_config.validation_split)

    # Model
    mlp_config: Config = config.model.mlp
    cnn_config: Config = config.model.cnn

    num_filters: list[int] = []
    kernel_sizes: list[int] = []
    for index, conv_layer in enumerate(cnn_config.conv_layers):
        num_filters.append(conv_layer[index].filters) # /!\ last filter number must be 1
        kernel_sizes.append(conv_layer[index].kernel_size) 
    
    model = DynamicModel(mlp_config.input_size, mlp_config.hidden_layers, num_filters, kernel_sizes, 1)

    # Training
    training_config: Config = config.training

    epochs: int = training_config.epochs
    initial_lr: float = training_config.initial_learning_rate

    optimizer_options: dict = {"adam": Adam, "sgd": SGD, "rmsprop": RMSprop}
    optimizer = optimizer_options[training_config.optimizer](model.parameters(), lr=initial_lr)

    criterion_options: dict = {"mse": MSELoss, "l1": L1Loss}
    criterion = criterion_options[training_config.loss]()

    if training_config.lr_scheduler.enabled:
        lr_scheduler_config: Config = training_config.lr_scheduler
        lr_scheduler_type = lr_scheduler_config.type
        if lr_scheduler_type == "plateau":
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_scheduler_config.factor, patience=lr_scheduler_config.patience)

    for repetition in range(config.experiment.repetitions):
        print(f"Starting repetition {repetition + 1} of experiment: {config.experiment.experiment_name}")

if __name__ == "__main__":
    pass