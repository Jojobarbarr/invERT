# configuration file for experiments

config = {
    "experiment": {
        "experiment_name": "None",  # Change this to the name of your experiment
        "repetitions": 5,  # Number of repetitions for the experiment
        "output_folder": "./results",  # Folder to save results
        "logging": {
            "save_checkpoints": True,  # Save model checkpoints
            "save_interval": 5,  # Interval (in epochs) to save checkpoints
            "save_best_model": True,  # Save the best model during training
            "log_interval": 10  # Logging interval during training
        }
    },

    "training": {
        "initial_learning_rate": 0.001,  # Initial learning rate for the optimizer
        "batch_size": 32,  # Batch size for training
        "epochs": 50,  # Number of training epochs
        "optimizer": "adam",  # Optimizer to use TODO: specify which optimizers are available
        "loss_function": "mean_squared_error",  # Loss function to use TODO: specify which loss functions are available
        "lr_scheduler": {
            "enabled": True,  # Enable learning rate scheduler
            "type": "step",  # Type of scheduler TODO: specify which schedulers are available
            "step_size": 10,  # Step size for scheduler
            "gamma": 0.5  # Decay rate for the scheduler
        }
    },

    "dataset": {
        "dataset_name": "custom_dataset",  # Name of the dataset
        "data_path": "./data",  # Path to the dataset
        "dataset_size": 10000,  # Total size of the dataset
        "validation_split": 0.2,  # Fraction of data for validation
        "input_shape": [64, 64, 1],  # Shape of the input data
    },
    
    "model": {
        "mlp": {
            "input_size": 3,  # Input size for the MLP
            "hidden_layers": [64, 128, 64],  # Hidden layer sizes
            "output_size": 32,  # Output size of the MLP
            "activation": "relu"  # Activation function
        },
        "cnn": {
            "input_channels": 1,  # Number of input channels
            "conv_layers": [
                {
                    "filters": 16,  # Number of filters
                    "kernel_size": 3,  # Size of the kernel
                    "stride": 1,  # Stride for the convolution
                    "padding": "same",  # Padding type
                    "activation": "relu"  # Activation function
                },
                {
                    "filters": 32,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": "same",
                    "activation": "relu"
                },
                {
                    "filters": 1,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": "same",
                    "activation": "sigmoid"
                }
            ]
        }
    }
}