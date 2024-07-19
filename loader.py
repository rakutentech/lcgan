import numpy as np
import os
import json
import random

import lc_gan
import torch
import torch.distributed as dist
from datetime import datetime
from torch.backends import cudnn
from utils.style_ops import grid_sample_gradfix
from utils.style_ops import conv2d_gradfix


def multi_gpu_setup(local_rank, args, gpus_per_node, port_number):
    cudnn.benchmark, cudnn.deterministic = True, False
    dist.init_process_group(backend='nccl',
                            init_method='tcp://%s:%s' % ('localhost', str(port_number)),
                            rank=local_rank,
                            world_size=gpus_per_node)
    torch.cuda.set_device(local_rank)
    # setting seeds
    random.seed(args.seed+local_rank)
    torch.manual_seed(args.seed+local_rank)
    torch.cuda.manual_seed_all(args.seed+local_rank)
    torch.cuda.manual_seed(args.seed+local_rank)
    np.random.seed(args.seed+local_rank)


def load_worker(local_rank, args, gpus_per_node, port_number):
    # setup multi-gpu processing
    multi_gpu_setup(local_rank, args, gpus_per_node, port_number)
    # Improves training speed
    conv2d_gradfix.enabled = True
    # Avoids errors with the augmentation pipe
    grid_sample_gradfix.enabled = True

    if args.phase == 'train':
        with open(os.path.join(args.model_name, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        gan_worker = lc_gan.LC_GAN(args, local_rank, gpus_per_node)
        epoch = 0
        
        if local_rank == 0:
            start_time = datetime.now()

        # Load the epoch number from epoch.txt if it exists
        epoch_file_path = os.path.join(args.model_name, 'epoch.txt')
        if os.path.exists(epoch_file_path):
            with open(epoch_file_path, "r") as file:
                epoch = int(file.read().strip()) + 1
                print("restart training from:", epoch)
            gan_worker.load_model()
            dist.barrier(gan_worker.group)
        
        while epoch <= args.epoch:
            gan_worker.requires_grad(gan_worker.generator, True)
            gan_worker.requires_grad(gan_worker.discriminator, False)
            g_loss = gan_worker.train_generator(epoch)
            gan_worker.ema_update(epoch)
            
            gan_worker.requires_grad(gan_worker.generator, False)            
            gan_worker.requires_grad(gan_worker.discriminator, True)
            if epoch >= args.freezeD_start:
                gan_worker.freeze_discriminator(args.freezeD_layer)            
            d_loss = gan_worker.train_discriminator(epoch)
            
            if epoch < 10000 and args.img_h > 256:
                gan_worker.train_dataset.apply_blur = True
            else:
                gan_worker.train_dataset.apply_blur = False

            if epoch % args.print_interval == 0:
                if local_rank == 0:
                    elapsed = datetime.now() - start_time
                    elapsed = str(elapsed).split(".")[0]
                    if epoch == 0:
                        file = open(os.path.join(args.model_name, 'log.txt'), "w")
                    else:
                        file = open(os.path.join(args.model_name, 'log.txt'), "a")
                    file.write("epoch:{loop}, elapsed:{elapsed}, "
                               "g_loss:{g_loss:.6f}, d_loss:{d_loss:.6f} \n"
                               .format(loop=epoch, elapsed=elapsed, g_loss=g_loss, d_loss=d_loss))
                    file.close()
                dist.barrier(gan_worker.group)
            
            if epoch % args.show_interval == 0:
                if local_rank == 0:
                    gan_worker.monitor_current_result(num_explore=20, w_psi=args.w_psi, epoch=epoch, images_per_output=args.geo_noise_dim)
                    gan_worker.feature_visualization(w_psi=args.w_psi)
                dist.barrier(gan_worker.group)

            if epoch % args.save_interval == 0 and epoch > 0:
                if local_rank == 0:
                    gan_worker.save_model()
                    with open(epoch_file_path, 'w') as f:
                        f.write(str(epoch))                              
                dist.barrier(gan_worker.group)

            if epoch % args.test_interval == 0:
                if epoch > 0:
                    fid_value = gan_worker.fid_evaluate()
                if local_rank == 0:
                    if epoch == 0:
                        file = open(os.path.join(args.model_name, 'fid.txt'), "w")
                    else:
                        gan_worker.save_best_model()
                        file = open(os.path.join(args.model_name, 'fid.txt'), "a")
                        file.write("epoch:{loop}, FID:{fid} \n".format(loop=epoch, fid=fid_value))
                    file.close()
                dist.barrier(gan_worker.group)
                                
            epoch += 1

    elif args.phase == 'fid_eval':
        # fid evaluation phase
        print(args)
        gan_worker = lc_gan.LC_GAN(args, local_rank, gpus_per_node)
        gan_worker.load_model()
        dist.barrier(gan_worker.group)
        fid_value = gan_worker.fid_evaluate()
        file = open(os.path.join(args.model_name, 'fid_({w_psi}).txt'.format(w_psi=args.w_psi)), "w")
        file.write("FID:{fid} \n".format(fid=fid_value))
        file.close()

    elif args.phase == 'fake_image_generation':
        print(args)
        gan_worker = lc_gan.LC_GAN(args, local_rank, gpus_per_node)
        gan_worker.load_model()
        dist.barrier(gan_worker.group)
        gan_worker.fake_image_generation(num_images=100)

    elif args.phase == 'geometry_shared_generation':
        print(args)
        gan_worker = lc_gan.LC_GAN(args, local_rank, gpus_per_node)
        gan_worker.load_model()
        dist.barrier(gan_worker.group)
        gan_worker.geometry_shared_generation(ctrl_dim=args.ctrl_dim, num_pairs=250)

    elif args.phase == 'appearance_shared_generation':
        print(args)
        gan_worker = lc_gan.LC_GAN(args, local_rank, gpus_per_node)
        gan_worker.load_model()
        dist.barrier(gan_worker.group)
        if local_rank == 0:
            gan_worker.monitor_current_result(num_explore=20, w_psi=args.w_psi, epoch=1112)        
        gan_worker.appearance_shared_generation(ctrl_dim=args.ctrl_dim, num_pairs=250)
        
    elif args.phase == 'joint_interpolation':
        print(args)
        gan_worker = lc_gan.LC_GAN(args, local_rank, gpus_per_node)
        gan_worker.load_model()
        dist.barrier(gan_worker.group)
        
        # gan_worker.joint_interpolation(dim1=18, dim2=21, dim3=32+19, dim4=32+3, num_images=10) # celeba_hq 512 [celeba_hq 512 : pitch, yaw, hair color, hair shape]
        gan_worker.joint_interpolation(dim1=22, dim2=21, dim3=32+16, dim4=10, num_images=10) # [celeba_hq 512 : smile, yaw, gender, zoom]
        # gan_worker.joint_interpolation(dim1=18, dim2=21, dim3=32+19, dim4=32+26, num_images=10)   # ffhq 512 
            
    elif args.phase == 'demo_generation':
        print(args)
        gan_worker = lc_gan.LC_GAN(args, local_rank, gpus_per_node)
        gan_worker.load_model()
        dist.barrier(gan_worker.group)
        for i in range(args.geo_noise_dim+args.app_noise_dim):
            print(i)
            gan_worker.demo_generation(controlled_dim=i, num_video=10)
            
    elif args.phase == 'latent_exploration':
        print(args)
        gan_worker = lc_gan.LC_GAN(args, local_rank, gpus_per_node)
        gan_worker.load_model()
        dist.barrier(gan_worker.group)
        gan_worker.latent_exploration(num_explore=6, num_images=4)            