# Training
train_steps = 3000000

# Data
batch_size = 256

# Optimizer
learning_rate = 0.0001

# Model
recurrent_steps = 8
feature_maps = 128
output_bits = 1  # does not seem to help

# Loss
objective_loss_scale = 0.001
logit_regularizer = 1e-8  # 1e-7 #does not seem to help
