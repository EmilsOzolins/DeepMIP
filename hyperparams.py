# Training
train_steps = 300000

# Data
batch_size = 64

# Optimizer
learning_rate = 0.0001

# Model
recurrent_steps = 8
feature_maps = 128
output_bits = 1 #does not seem to help

# Loss
objective_loss_scale = 0.03
logit_regularizer = 0# 1e-7 #does not seem to help
