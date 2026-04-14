# Baseline configuration
out_dir = 'out-taylor-max-iters-850'
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False
wandb_log = False
wandb_project = 'nanoGPT-assignment'
wandb_run_name = 'taylor-max-iters-850'
dataset = 'taylor_swift_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
learning_rate = 1e-3
max_iters = 850
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
weight_decay = 1e-1
device = 'mps' # change to 'cuda' if you have a GPU # changed from CPU
compile = False 
# set True only on Linux with GPU