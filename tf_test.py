import tensorflow as tf
print('Is TF build with CUDA: ', tf.test.is_built_with_cuda())
# print(tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None))
print(tf.config.list_physical_devices('GPU'))
print('TF Version:', tf.__version__)