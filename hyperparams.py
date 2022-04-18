# Training
train_steps = 3000000

# Data
batch_size = 2

# Optimizer
learning_rate = 0.0001

# Model
recurrent_steps = 2
feature_maps = 64
output_bits = 1  # does not seem to help

# Loss
objective_loss_scale = 0.03
logit_regularizer = 1e-4  # 1e-7 #does not seem to help
