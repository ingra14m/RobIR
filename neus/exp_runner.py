import functools
import shutil
import os

import torch.random
import torch
import numpy as np
from absl import app, flags
from optimization.trainer import Trainer
from model.neus_fields import NeuSModel
import gin

# Define the flags used by both train.py and eval.py
flags.DEFINE_multi_string('gin_file', None,
                          'List of paths to the config files.')
flags.DEFINE_multi_string('gin_param', None,
                          'Newline separated list of Gin parameter bindings.')
# flags.DEFINE_string('train_dir', None, 'where to store ckpts and logs')
# flags.DEFINE_string('data_dir', None, 'input data directory.')
flags.DEFINE_boolean('test', None, 'just test the model and generate a video')
flags.DEFINE_boolean('new', None, 'do not resume from the checkpoints')
flags.DEFINE_integer("gpu_index", 0, "Index of GPU")


def parse_gin_file():
    gin.parse_config_files_and_bindings(flags.FLAGS.gin_file,
                                        flags.FLAGS.gin_param)
    return flags.FLAGS.gin_file


def main(unused_argv):
    gin_files = parse_gin_file()

    torch.random.manual_seed(20200823)
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    np.random.seed(20201473)

    torch.cuda.set_device(flags.FLAGS.gpu_index)
    print("Current CUDA device = {}".format(torch.cuda.current_device()))

    trainer = Trainer(resume=not flags.FLAGS.new)

    for gin_file in gin_files:
        shutil.copy(gin_file,
                    trainer.logger.path_of(os.path.basename(gin_file)))

    if not flags.FLAGS.test:
        trainer.train()

    trainer.test()


if __name__ == '__main__':
    app.run(main)
