import argparse
import os
import random
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import loader
import copy
import json


def parse_args():
    desc = "Pytorch implementation of LC-GAN"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--phase', type=str, default='train', help='train, evaluation, or ...')
    parser.add_argument("--best", default=False, action="store_true", help='Load the best model')

    parser.add_argument('--tau', type=float, default=0.05, help='The margin of contrastive loss')
    parser.add_argument('--l_adv', type=float, default=1.0, help='The weight of adversarial loss')
    parser.add_argument('--l_aux', type=float, default=0.5, help='The weight of loss in auxiliary mapping')
    parser.add_argument('--l_r1', type=float, default=10.0, help='The weight r1 regularization')
    parser.add_argument('--l_s', type=float, default=0.0000001, help='The weight of sparsity regularization')
    
    parser.add_argument('--max_flow_scale', type=float, default=0.1, help='maximum flow scale')
    parser.add_argument('--geo_noise_dim', type=int, default=64, help='length of noise dimension')
    parser.add_argument('--app_noise_dim', type=int, default=64, help='length of noise dimension')
    parser.add_argument('--geo_projection_dim', type=int, default=256, help='length of projected dimension')
    parser.add_argument('--app_projection_dim', type=int, default=256, help='length of projected dimension')
    parser.add_argument('--geo_latent_dim', type=int, default=512, help='length of intermediate latent dimension')
    parser.add_argument('--app_latent_dim', type=int, default=64, help='length of intermediate latent dimension')

    parser.add_argument('--epoch', type=int, default=100000, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size')
    parser.add_argument('--g_lr', type=float, default=0.002, help='The learning rate of the generator')
    parser.add_argument('--d_lr', type=float, default=0.002, help='The learning rate of the discriminator')
    parser.add_argument('--beta1', type=float, default=0.0, help='The beta1 of ADAM optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='The beta2 of ADAM optimizer')
    parser.add_argument('--g_ema_decay', type=float, default=0.9999, help='decaying rate of EMA')
    parser.add_argument('--g_ema_start', type=int, default=0, help='start step of applying EMA')
    parser.add_argument('--freezeD_start', type=int, default=100000, help='start step of applying freezeD')
    parser.add_argument('--freezeD_layer', type=int, default=5, help='first n layers of applying freezeD')

    parser.add_argument('--img_resolution', type=int, default=256, help='The size of image resolution')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--psi', type=float, default=2.0, help='The truncation value of noise vector')
    parser.add_argument('--w_psi', type=float, default=1.0, help='The truncation value of latent vector')

    parser.add_argument('--dataset_path', type=str, default='./', help='dataset_name')
    parser.add_argument('--model_name', type=str, default='', help='model name')
    parser.add_argument('--save_dir', type=str, default='model', help='Directory name to save the model')
    parser.add_argument('--sample_dir', type=str, default='samples', help='Directory name to save the training results')

    parser.add_argument('--num_fakes', type=int, default=10, help='Number of images to generate')
    parser.add_argument('--ctrl_dim', type=int, default=-1, help='control dimension')
    parser.add_argument('--num_videos', type=int, default=10, help='Number of videos to generate')

    parser.add_argument("--save_interval", type=int, default=5000, help="save interval")
    parser.add_argument("--print_interval", type=int, default=100, help="print interval")
    parser.add_argument('--show_interval', type=int, default=1000, help='interval of showing images in training')
    return check_args(parser.parse_args())


def check_folder(test_dir):
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    return test_dir


def check_args(args):
    # --model_name
    try:
        assert bool(args.model_name)
    except:
        print('model name must be given')

    check_folder(args.model_name)

    # --save dir
    check_folder(os.path.join(args.model_name, args.save_dir))
    check_folder(os.path.join(args.model_name, args.sample_dir))

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


def main():
    # parse arguments
    print("Checking arguments...")
    args = parse_args()
    print(args)

    # print arguments as a file args.txt
    gpus_per_node, rank = torch.cuda.device_count(), torch.cuda.current_device()
    port_number = random.randint(22000, 23000)

    print("Processing with {num_gpus} GPUs".format(num_gpus=gpus_per_node))
    mp.set_start_method("spawn", force=True)
    try:
        torch.multiprocessing.spawn(fn=loader.load_worker,
                                    args=(args, gpus_per_node, port_number),
                                    nprocs=gpus_per_node)
    except KeyboardInterrupt:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
