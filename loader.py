import numpy as np
import os
import json
import random

import worker
import torch
import torch.distributed as dist
from datetime import datetime
from torch.backends import cudnn


def multi_gpu_setup(local_rank, args, gpus_per_node, port_number):
    cudnn.benchmark, cudnn.deterministic = True, False
    dist.init_process_group(backend='nccl',
                            init_method='tcp://%s:%s' % ('localhost', str(port_number)),
                            rank=local_rank,
                            world_size=gpus_per_node)
    torch.cuda.set_device(local_rank)


def load_worker(local_rank, args, gpus_per_node, port_number):
    # setup multi-gpu processing
    multi_gpu_setup(local_rank, args, gpus_per_node, port_number)

    if args.phase == 'train':
        with open(os.path.join(args.model_name, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        gan_worker = worker.WORKER(args, local_rank, gpus_per_node)
        epoch = 0
        
        start_time = datetime.now()

        # Load the epoch number from epoch.txt if it exists
        epoch_file_path = os.path.join(args.model_name, 'epoch.txt')
        if os.path.exists(epoch_file_path):
            with open(epoch_file_path, "r") as file:
                epoch = int(file.read().strip()) + 1
                print("restart training from:", epoch)
            gan_worker.load_model()
            if epoch > args.freezeD_start:
                gan_worker.drop_learning_rate()
                print("drop_learning_rate from here")            
            dist.barrier(gan_worker.group)
        
        while epoch <= args.epoch:
            if epoch == args.freezeD_start:
                gan_worker.drop_learning_rate()
            gan_worker.requires_grad(gan_worker.generator, True)
            gan_worker.requires_grad(gan_worker.discriminator, False)
            g_loss = gan_worker.train_generator(epoch)
            gan_worker.ema_update(epoch)
            
            gan_worker.requires_grad(gan_worker.generator, False)            
            gan_worker.requires_grad(gan_worker.discriminator, True)
            if epoch >= args.freezeD_start:
                gan_worker.freeze_discriminator(args.freezeD_layer)            
            d_loss = gan_worker.train_discriminator(epoch)

            if epoch % args.print_interval == 0:
                elapsed = datetime.now() - start_time
                if local_rank == 0:
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
            
            if epoch % args.show_interval == 0 and epoch > 0:
                if local_rank == 0:
                    gan_worker.monitor_current_result(num_explore=20, w_psi=args.w_psi, epoch=epoch, images_per_output=args.geo_noise_dim)
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
        gan_worker = worker.WORKER(args, local_rank, gpus_per_node)
        gan_worker.load_model()
        dist.barrier(gan_worker.group)
        fid_value = gan_worker.fid_evaluate()
        file = open(os.path.join(args.model_name, 'fid_({w_psi}).txt'.format(w_psi=args.w_psi)), "w")
        file.write("FID:{fid} \n".format(fid=fid_value))
        file.close()

    elif args.phase == 'fake_image_generation':
        gan_worker = worker.WORKER(args, local_rank, gpus_per_node)
        gan_worker.load_model()
        dist.barrier(gan_worker.group)
        gan_worker.fake_image_generation(num_images=100)
