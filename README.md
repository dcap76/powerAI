# step 1 
compile so file:
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# update parser.py
function load_data add sep='|'




# json config
# fen lei
{
    "batch_size": 64,
    "dropout_keep_prob": 0.5,
    "embedding_dim": 300,
    "evaluate_every": 100,
    "filter_sizes": "3,4,5",
    "hidden_unit": 300,
    "l2_reg_lambda": 0.0,
    "max_pool_size": 4,
    "non_static": true,
    "num_epochs": 50,
    "num_filters": 128
}

# qinggan
{
    "batch_size": 128,
    "dropout_keep_prob": 0.5,
    "embedding_dim": 300,
    "evaluate_every": 100,
    "filter_sizes": "3,4,5,6",
    "hidden_unit": 300,
    "l2_reg_lambda": 0.0,
    "max_pool_size": 2,
    "non_static": true,
    "num_epochs": 50,
    "num_filters": 128
}

#version huang:
#python word2vec_kernel.py --train_data=./data/text8.txt --eval_data=./data/questions-words2.txt --save_path=./data/


# step 2
python train.py

# use GPU 
# vi text_cnn_rnn.py



# step 3 
python predict.py ./trained_results_xxxxxxxxxx/
