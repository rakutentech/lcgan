import torch
import torch.nn.functional as F
import numpy as np
import cnn
import custom_dataset
import loss
import os
import torch.distributed as dist
import copy
import glob
import math
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.nn import DataParallel
from torchvision.utils import save_image, make_grid
from ema import Ema
from scipy.stats import truncnorm

from eval.inception import InceptionV3
import eval.fid as fid

from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import resize, InterpolationMode
import av

class WORKER(object):
    def __init__(self, args, local_rank, gpus_per_node):
        self.args = args
        self.local_rank = local_rank
        self.gpus_per_node = gpus_per_node
        self.local_batch_size = args.batch_size // gpus_per_node
        self.global_iter_counter = 0
        self.num_dataloading_workers = 4
        self.train_dataloader, self.train_dataset, self.train_iter = self.prepare_training_dataset()
        self.generator, self.discriminator, self.g_optimizer, self.d_optimizer = self.set_cnn_models()
        self.generator_ema = copy.deepcopy(self.generator)
        self.ema = Ema(self.generator, self.generator_ema, self.args.g_ema_decay, self.args.g_ema_start)
        self.best_fid = 9999
        self.group = dist.new_group([n for n in range(self.gpus_per_node)])

    def prepare_training_dataset(self):
        # Prepare training dataset
        if self.local_rank == 0:
            print("Load training dataset")

        train_dataset = custom_dataset.Dataset_(self.args.dataset_path, self.args.img_resolution,
                                                self.local_batch_size, self.args.phase == 'train')
        if self.local_rank == 0:
            print("Train dataset size: {dataset_size}".format(dataset_size=len(train_dataset)))

        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=self.gpus_per_node,
                                           rank=self.local_rank,
                                           shuffle=True,
                                           drop_last=True)

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=self.local_batch_size,
                                      shuffle=(train_sampler is None),
                                      pin_memory=True,
                                      num_workers=self.num_dataloading_workers,
                                      sampler=train_sampler,
                                      drop_last=True,
                                      persistent_workers=True)

        train_dataloader.sampler.set_epoch(self.global_iter_counter)
        train_iter = iter(train_dataloader)
        return train_dataloader, train_dataset, train_iter

    def set_cnn_models(self):
        generator = cnn.Generator(self.args).to(self.local_rank)
        discriminator = cnn.Discriminator(self.args).to(self.local_rank)
        if self.local_rank == 0:
            print(discriminator)
            print(generator)

        g_parameters, d_parameters = [], []
        for g_name, g_param in generator.named_parameters():
            g_parameters.append(g_param)
        for d_name, d_param in discriminator.named_parameters():
            d_parameters.append(d_param)

        generator = DistributedDataParallel(generator,
                                            device_ids=[self.local_rank],
                                            broadcast_buffers=False,
                                            find_unused_parameters=True)
        
        discriminator = DistributedDataParallel(discriminator,
                                                device_ids=[self.local_rank],
                                                broadcast_buffers=False,
                                                find_unused_parameters=True)

        betas_g = [self.args.beta1, self.args.beta2]
        betas_d = [self.args.beta1, self.args.beta2]
        eps_ = 1e-8

        g_optimizer = torch.optim.Adam(params=g_parameters,
                                       lr=self.args.g_lr,
                                       betas=betas_g,
                                       eps=eps_)

        d_optimizer = torch.optim.Adam(params=d_parameters,
                                       lr=self.args.d_lr,
                                       betas=betas_d,
                                       eps=eps_)

        return generator, discriminator, g_optimizer, d_optimizer

    def sample_data_basket(self):
        try:
            image, geometry_change, appearance_change = next(self.train_iter)
        except StopIteration:
            self.global_iter_counter += 1
            if self.args.phase == 'train':
                self.train_dataloader.sampler.set_epoch(self.global_iter_counter)
            else:
                pass
            self.train_iter = iter(self.train_dataloader)
            image, geometry_change, appearance_change = next(self.train_iter)
        return image, geometry_change, appearance_change

    def freeze_discriminator(self, freeze_up_to_index=3):
        for i, (name, layer) in enumerate(self.discriminator.module.shared_model.named_children()):
            if i <= freeze_up_to_index:
                for param in layer.parameters():
                    param.requires_grad = False
                print(f"Freezing layer {i}: {layer}")

    def drop_learning_rate(self):
        new_g_lr = self.args.g_lr * 0.5
        new_d_lr = self.args.d_lr * 0.5
        betas_g = [self.args.beta1, self.args.beta2]
        betas_d = [self.args.beta1, self.args.beta2]
        eps_ = 1e-8

        g_parameters = list(self.generator.module.parameters())
        d_parameters = list(self.discriminator.module.parameters())

        self.g_optimizer = torch.optim.Adam(params=g_parameters,
                                            lr=new_g_lr,
                                            betas=betas_g,
                                            eps=eps_)

        self.d_optimizer = torch.optim.Adam(params=d_parameters,
                                            lr=new_d_lr,
                                            betas=betas_d,
                                            eps=eps_)
        
    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def train_discriminator(self, epoch):
        self.d_optimizer.zero_grad()
        
        image, geometry_change, appearance_change = self.sample_data_basket()
        image = image.to(self.local_rank, non_blocking=True).cuda()
        geometry_change = geometry_change.to(self.local_rank, non_blocking=True).cuda()
        appearance_change = appearance_change.to(self.local_rank, non_blocking=True).cuda()

        rand1 = torch.randn(self.local_batch_size, self.args.geo_noise_dim, device=self.local_rank)
        rand2 = torch.randn(self.local_batch_size, self.args.app_noise_dim, device=self.local_rank)
        
        fake_img = self.generator(rand1, rand2)
        fake_logit, _, _ = self.discriminator(fake_img, False)
        
        if epoch % 2 == 1:
            image.requires_grad_(True)
            real_logit, _, _ = self.discriminator(image, False)
            real_label = torch.ones(self.local_batch_size, 1, device=self.local_rank) * 0.95
            fake_label = torch.zeros(self.local_batch_size, 1, device=self.local_rank)
            real_loss = F.binary_cross_entropy_with_logits(real_logit, real_label)
            fake_loss = F.binary_cross_entropy_with_logits(fake_logit, fake_label)            
            d_adv_loss = real_loss + fake_loss
            r1_loss = loss.cal_r1_reg(real_logit, image, self.local_rank) * self.args.l_r1
            d_loss = d_adv_loss + r1_loss
        else:
            real_logit, geometry_feat, appearance_feat = self.discriminator(image, True)
            _, geometry_positive, appearance_negative = self.discriminator(geometry_change, True)
            _, geometry_negative, appearance_positive = self.discriminator(appearance_change, True)
            real_label = torch.ones(self.local_batch_size, 1, device=self.local_rank) * 0.95
            fake_label = torch.zeros(self.local_batch_size, 1, device=self.local_rank)
            real_loss = F.binary_cross_entropy_with_logits(real_logit, real_label)
            fake_loss = F.binary_cross_entropy_with_logits(fake_logit, fake_label)            
            d_adv_loss = real_loss + fake_loss
            d_aug_loss = (loss.contrastive_loss(geometry_feat, geometry_positive, geometry_negative, self.args.tau) 
                        + loss.contrastive_loss(appearance_feat, appearance_positive, appearance_negative, self.args.tau)) * self.args.l_aux
            d_loss = d_adv_loss + d_aug_loss
            
        d_loss.backward()
        self.d_optimizer.step()
        return d_loss.item()
    
    def train_generator(self, epoch):
        self.g_optimizer.zero_grad()

        rand1 = torch.randn(self.local_batch_size, self.args.geo_noise_dim, device=self.local_rank)
        rand2 = torch.randn(self.local_batch_size, self.args.app_noise_dim, device=self.local_rank)
        resample1 = torch.randn(self.local_batch_size, self.args.geo_noise_dim, device=self.local_rank)
        resample2 = torch.randn(self.local_batch_size, self.args.app_noise_dim, device=self.local_rank)

        if epoch % 2 == 1:
            anchor_image = self.generator(rand1, rand2)
            logit, _, _ = self.discriminator(anchor_image, False)
            real_label = torch.ones(self.local_batch_size, 1, device=self.local_rank)
            g_adv_loss = F.binary_cross_entropy_with_logits(logit, real_label)
            g_loss = g_adv_loss
        else:
            anchor_image  = self.generator(rand1, rand2)
            resample_geometry = self.generator(resample1, rand2)
            resample_appearance = self.generator(rand1, resample2)
            
            logit, geometry_feat, appearance_feat = self.discriminator(anchor_image, True)
            _, geometry_positive, appearance_negative = self.discriminator(resample_geometry, True)
            _, geometry_negative, appearance_positive = self.discriminator(resample_appearance, True)

            real_label = torch.ones(self.local_batch_size, 1, device=self.local_rank)
            g_adv_loss = F.binary_cross_entropy_with_logits(logit, real_label)
            g_aug_loss = (loss.contrastive_loss(geometry_feat, geometry_positive, geometry_negative, self.args.tau) 
                        + loss.contrastive_loss(appearance_feat, appearance_positive, appearance_negative, self.args.tau)) * self.args.l_aux

            diagonal_params1 = self.generator.module.geometry_mapping.diagonal_params.view(-1)
            diagonal_params2 = self.generator.module.appearance_mapping.diagonal_params.view(-1)
            g_sparsity_loss = torch.norm(torch.cat([diagonal_params1, diagonal_params2]), p=1) * self.args.l_s
            g_loss = g_adv_loss + g_aug_loss + g_sparsity_loss
        
        g_loss.backward()
        self.g_optimizer.step()
        return g_loss.item()

    def ema_update(self, current_step):
        self.ema.update(current_step)

    def save_model(self):
        print('save model')
        save_path = os.path.join(self.args.model_name, self.args.save_dir)
        generator_path = '{}/gen_model.ckpt'.format(save_path)
        generator_ema_path = '{}/gen_ema_model.ckpt'.format(save_path)
        discriminator_path = '{}/disc_model.ckpt'.format(save_path)
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.generator_ema.state_dict(), generator_ema_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)

    def save_best_model(self):
        print('save best model')
        save_path = os.path.join(self.args.model_name, self.args.save_dir)
        generator_path = '{}/gen_model_best.ckpt'.format(save_path)
        generator_ema_path = '{}/gen_ema_model_best.ckpt'.format(save_path)
        discriminator_path = '{}/disc_model_best.ckpt'.format(save_path)
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.generator_ema.state_dict(), generator_ema_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)

    def load_model(self):
        print('load model')
        load_path = os.path.join(self.args.model_name, self.args.save_dir)
        if self.args.best:
            generator_path = '{}/gen_model_best.ckpt'.format(load_path)
            generator_ema_path = '{}/gen_ema_model_best.ckpt'.format(load_path)
            discriminator_path = '{}/disc_model_best.ckpt'.format(load_path)
        else:
            generator_path = '{}/gen_model.ckpt'.format(load_path)
            generator_ema_path = '{}/gen_ema_model.ckpt'.format(load_path)
            discriminator_path = '{}/disc_model.ckpt'.format(load_path)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
        self.generator.load_state_dict(torch.load(generator_path, map_location=map_location))
        self.generator_ema.load_state_dict(torch.load(generator_ema_path, map_location=map_location))
        self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=map_location))

    def monitor_current_result(self, num_explore=10, w_psi=0.7, epoch=0, nrow=8, images_per_output=32):
        disp_resolution = 128
        to_pil = ToPILImage()
        for i in range(self.args.geo_noise_dim//images_per_output):
            mult_frames = []
            for ii in range(5):
                frames = []
                geometry_start = torch.randn(images_per_output, self.args.geo_noise_dim, device=self.local_rank)
                geometry_end = geometry_start.clone()
                appearance_code = torch.randn(images_per_output, self.args.app_noise_dim, device=self.local_rank)

                # Modify the diagonal elements for each sample
                for j in range(images_per_output):
                    idx = i * images_per_output + j
                    geometry_start[j, idx] = -self.args.psi
                    geometry_end[j, idx] = self.args.psi
                
                for j in range(num_explore):
                    canvas = []
                    inter_code = geometry_start.lerp(geometry_end, 1/(num_explore)*j)
                    for k in range(images_per_output//self.local_batch_size):
                        with torch.no_grad():
                            geometry_change = self.generator_ema(
                                inter_code[k*self.local_batch_size:(k+1)*self.local_batch_size,:],
                                appearance_code[k*self.local_batch_size:(k+1)*self.local_batch_size,:],
                                w_psi
                            )
                        canvas.append(geometry_change)
                    canvas = torch.cat(canvas, dim=0)
                    canvas = make_grid(canvas, nrow=nrow, padding=0)
                    canvas = ((canvas + 1) / 2).clamp(0.0, 1.0)
                    canvas = resize(canvas, size=(disp_resolution * images_per_output // nrow, disp_resolution * nrow), interpolation=InterpolationMode.BILINEAR)
                    frames.append(to_pil(canvas))

                for j in range(num_explore):
                    canvas = []
                    inter_code = geometry_end.lerp(geometry_start, 1/(num_explore)*j)
                    for k in range(images_per_output//self.local_batch_size):
                        with torch.no_grad():
                            geometry_change = self.generator_ema(
                                inter_code[k*self.local_batch_size:(k+1)*self.local_batch_size,:],
                                appearance_code[k*self.local_batch_size:(k+1)*self.local_batch_size,:],
                                w_psi
                            )
                        canvas.append(geometry_change)
                    canvas = torch.cat(canvas, dim=0)
                    canvas = make_grid(canvas, nrow=nrow, padding=0)
                    canvas = ((canvas + 1) / 2).clamp(0.0, 1.0)
                    canvas = resize(canvas, size=(disp_resolution * images_per_output // nrow, disp_resolution * nrow), interpolation=InterpolationMode.BILINEAR)
                    frames.append(to_pil(canvas))

                # repeat frames 2 times 
                mult_frames.extend(frames * 2)
            save_name = os.path.join(self.args.model_name, "samples/geometry_{num}_{b}.mp4".format(num=epoch,b=i))
            self.save_mp4_video(mult_frames, save_name, fps=15)
            
        # appearance 
        for i in range(self.args.app_noise_dim//images_per_output):
            mult_frames = []
            for ii in range(5):
                frames = []
                appearance_start = torch.randn(images_per_output, self.args.geo_noise_dim, device=self.local_rank)
                appearance_end = appearance_start.clone()
                geometry_code = torch.randn(images_per_output, self.args.app_noise_dim, device=self.local_rank)

                # Modify the diagonal elements for each sample
                for j in range(images_per_output):
                    idx = i * images_per_output + j
                    appearance_start[j, idx] = -self.args.psi
                    appearance_end[j, idx] = self.args.psi
                
                for j in range(num_explore):
                    canvas = []
                    inter_code = appearance_start.lerp(appearance_end, 1/(num_explore)*j)
                    for k in range(images_per_output//self.local_batch_size):
                        with torch.no_grad():
                            appearance_change = self.generator_ema(
                                geometry_code[k*self.local_batch_size:(k+1)*self.local_batch_size,:],
                                inter_code[k*self.local_batch_size:(k+1)*self.local_batch_size,:],
                                w_psi
                            )
                        canvas.append(appearance_change)
                    canvas = torch.cat(canvas, dim=0)
                    canvas = make_grid(canvas, nrow=nrow, padding=0)
                    canvas = ((canvas + 1) / 2).clamp(0.0, 1.0)
                    canvas = resize(canvas, size=(disp_resolution * images_per_output // nrow, disp_resolution * nrow), interpolation=InterpolationMode.BILINEAR)
                    frames.append(to_pil(canvas))

                for j in range(num_explore):
                    canvas = []
                    inter_code = appearance_end.lerp(appearance_start, 1/(num_explore)*j)
                    for k in range(images_per_output//self.local_batch_size):
                        with torch.no_grad():
                            appearance_change = self.generator_ema(
                                geometry_code[k*self.local_batch_size:(k+1)*self.local_batch_size,:],
                                inter_code[k*self.local_batch_size:(k+1)*self.local_batch_size,:],
                                w_psi
                            )
                        canvas.append(appearance_change)
                    canvas = torch.cat(canvas, dim=0)                 
                    canvas = make_grid(canvas, nrow=nrow, padding=0)
                    canvas = ((canvas + 1) / 2).clamp(0.0, 1.0)
                    canvas = resize(canvas, size=(disp_resolution * images_per_output // nrow, disp_resolution * nrow), interpolation=InterpolationMode.BILINEAR)
                    frames.append(to_pil(canvas))
                    
                # repeat frames 2 times 
                mult_frames.extend(frames * 2)                    
            save_name = os.path.join(self.args.model_name, "samples/appearance_{num}_{b}.mp4".format(num=epoch,b=i))
            self.save_mp4_video(mult_frames, save_name, fps=15)
            
    def save_mp4_video(self, frames, save_path, fps):
        width, height = frames[0].size
        output = av.open(save_path, 'w')
        stream = output.add_stream('libx264', rate=fps)
        stream.width = width
        stream.height = height
        stream.open()
        for frame in frames:
            frame_np = np.array(frame)
            video_frame = av.VideoFrame.from_ndarray(frame_np, format='rgb24')
            packet = stream.encode(video_frame)
            output.mux(packet)
        
        output.mux(stream.encode())
        output.close()
        
    def fid_evaluate(self):
        inception = InceptionV3([3], normalize_input=False).to(self.local_rank)
        inception.eval()
        
        num_generate = len(self.train_dataloader.dataset)
        if num_generate > 50000:
            num_generate = 50000
        num_batches = int(math.floor(float(num_generate) / float(self.local_batch_size)))

        training_features = []
        for i in tqdm(range(num_batches), disable=self.local_rank != 0):
            if self.args.phase == 'train':
                image, _, _ = self.sample_data_basket()
            else:
                image, _ = next(self.train_iter)
            with torch.no_grad():
                image = image.to(self.local_rank, non_blocking=True).cuda()
                feature = inception(image)[0].view(image.shape[0], -1)
                training_features.append(feature.to("cpu"))
        
        gen_features = []
        for i in tqdm(range(num_batches), disable=self.local_rank != 0):
            geometry_code = torch.randn(self.local_batch_size, self.args.geo_noise_dim, device=self.local_rank)
            appearance_code = torch.randn(self.local_batch_size, self.args.app_noise_dim, device=self.local_rank)
            with torch.no_grad():
                fake_images = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                feat = inception(fake_images)[0].view(fake_images.shape[0], -1)
                gen_features.append(feat.to("cpu"))

        gen_features = torch.cat(gen_features, 0).numpy()
        print(gen_features.shape)
        sample_mean = np.mean(gen_features, 0)
        sample_cov = np.cov(gen_features, rowvar=False)

        training_features = torch.cat(training_features, 0).numpy()
        print(training_features.shape)
        real_mean = np.mean(training_features, 0)
        real_cov = np.cov(training_features, rowvar=False)

        fid_value = fid.calc_fid(sample_mean, sample_cov, real_mean, real_cov)
        print("fid_value:", fid_value)
        if fid_value < self.best_fid and self.args.phase == 'train':
            self.best_fid = fid_value

        return fid_value


    def fake_image_generation(self, num_images=50):
        count = 0
        for ns in tqdm(range(num_images), disable=self.local_rank != 0):
            geometry_code = torch.randn(self.local_batch_size, self.args.geo_noise_dim, device=self.local_rank)
            appearance_code = torch.randn(self.local_batch_size, self.args.app_noise_dim, device=self.local_rank)
            with torch.no_grad():
                fake_images = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
            
            fake_images = ((fake_images + 1) / 2).clamp(0.0, 1.0)
            folder_path = os.path.join(self.args.model_name, 'fakes')
            save_path = os.path.join(folder_path, "{num:04d}_images.jpg".format(num=count))
            save_image(fake_images, save_path, padding=0, nrow=1)
            count = count + 1
            

    def check_folder(self, test_dir):
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

                    
    def demo_generation(self, controlled_dim=0, num_video=1, num_explore=30, num_repeat=1):
        folder_path = os.path.join(self.args.model_name, 'demo')
        self.check_folder(folder_path)
        to_pil = ToPILImage()
        for n in range(num_video):
            mult_frames = []
            frames = []
            latent_code = torch.randn(self.local_batch_size, self.args.geo_noise_dim + self.args.app_noise_dim, device=self.local_rank)
            interval = self.args.psi*2.0/(num_explore)
            latent_code[:,controlled_dim] = -self.args.psi - interval
            
            for i in range(num_explore):
                latent_code[:,controlled_dim] = latent_code[:,controlled_dim] + interval
                geometry_code, appearance_code = torch.chunk(latent_code, chunks=2, dim=1)
                with torch.no_grad():
                    image = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                image = ((image + 1) / 2).clamp(0.0, 1.0)
                                
                n_rows = int(self.local_batch_size ** 0.5)
                canvas = make_grid(image, nrow=n_rows, padding=0)
                canvas = canvas.clamp(0.0, 1.0)
                frames.append(to_pil(canvas))
                
            for i in range(num_explore):
                latent_code[:,controlled_dim] = latent_code[:,controlled_dim] - interval
                geometry_code, appearance_code = torch.chunk(latent_code, chunks=2, dim=1)
                with torch.no_grad():
                    image = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                image = ((image + 1) / 2).clamp(0.0, 1.0)
                                
                n_rows = int(self.local_batch_size ** 0.5)
                canvas = make_grid(image, nrow=n_rows, padding=0)
                canvas = canvas.clamp(0.0, 1.0)
                frames.append(to_pil(canvas))

            # repeat frames 2 times 
            mult_frames.extend(frames * num_repeat)
            save_name = os.path.join(self.args.model_name, "demo/controlled_dim={num}_{n}.mp4".format(num=controlled_dim,n=n))
            self.save_mp4_video(mult_frames, save_name, fps=num_explore)
