# Define your sweep configuration
regular_baseline_config = {
    'method': 'grid',  # or 'random', 'bayes'
    'metric': {
        'name': 'val_loss',  # or whatever metric you want to optimize
        'goal': 'minimize'  # or 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-5, 5e-4, 1e-4, 5e-3, 1e-3]  # Only these learning rates
        },
        'batch_size': {
            'values': [32, 64, 128, 256]  # Only these batch sizes
        }
    }
}