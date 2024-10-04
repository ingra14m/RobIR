import argparse
from training.train_cesr import ClusteredAlbedoTrainRunner
from training.train_visibility import VisTrainRunner
from training.train_pbr import PBRTrainRunner
from training.train_normal import NormalTrainRunner

render_dict = {
    'CESR': ClusteredAlbedoTrainRunner,
    'Vis': VisTrainRunner,
    'PBR': PBRTrainRunner,
    'Norm': NormalTrainRunner
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='')
    parser.add_argument('--exps_folder_name', type=str, default='exps')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--trainstage', type=str, default='IDR', help='')

    parser.add_argument('--data_split_dir', type=str, default='')
    parser.add_argument('--frame_skip', type=int, default=1, help='skip frame when training')

    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--max_niter', type=int, default=200001, help='max number of iterations to train for')
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint epoch number of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--index', type=int, default=0, help="plot index")
    parser.add_argument('--plot_only', default=False, action="store_true",
                        help='If set, just plot in the running stage.')
    parser.add_argument('--chunk', type=int, default=1024, help='chunk size in order to avoid OOM.')

    opt = parser.parse_args()

    from confs_sg import env_path

    env_path.set_path(opt.expname)

    trainrunner = render_dict[opt.trainstage](conf=opt.conf,
                                              exps_folder_name=opt.exps_folder_name,
                                              expname=opt.expname,
                                              data_split_dir=opt.data_split_dir,
                                              frame_skip=opt.frame_skip,
                                              batch_size=opt.batch_size,
                                              max_niters=opt.max_niter,
                                              is_continue=opt.is_continue,
                                              timestamp=opt.timestamp,
                                              checkpoint=opt.checkpoint,
                                              index=opt.index,
                                              plot_only=opt.plot_only,
                                              chunk=opt.chunk)

    trainrunner.run()
