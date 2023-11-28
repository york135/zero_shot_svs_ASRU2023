# Mel
num_mels = 80
num_spec = 80

# FastSpeech
vocab_size = 300
pitch_size = 100
max_seq_len = 3000

pitch_encoder_n_layer = 2

encoder_dim = 256
encoder_n_layer = 2
encoder_head = 2
encoder_conv1d_filter_size = 1024

decoder_dim = 256
decoder_n_layer = 2
decoder_head = 2
decoder_conv1d_filter_size = 1024

fft_conv1d_kernel = (3, 1)
fft_conv1d_padding = (1, 0)

dropout = 0.1

# Train
checkpoint_path = "./models/model_test"
logger_path = "./loggers/logger_test"

total_step = 300000
vali_step = 1000
n_warm_up_step = 100
epochs = 100

batch_expand_size = 4
batch_size = 1

learning_rate = 1e-5
weight_decay = 1e-6
grad_clip_thresh = 1.0

