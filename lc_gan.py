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
from utils.ema import Ema
from scipy.stats import truncnorm

from eval.inception import InceptionV3
import eval.fid as fid

from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import resize, InterpolationMode
import av

class LC_GAN(object):
    def __init__(self, args, local_rank, gpus_per_node):
        self.args = args
        self.local_rank = local_rank
        self.gpus_per_node = gpus_per_node
        self.local_batch_size = args.batch_size // gpus_per_node
        self.global_iter_counter = 0
        self.num_dataloading_workers = 4
        # if self.args.phase == 'train':
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

        train_dataset = custom_dataset.Dataset_(self.args.dataset_path, self.args.img_w,
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

    def freeze_discriminator(self, freeze_up_to_index=5):
        for d_name, d_param in self.discriminator.module.shared_model.named_parameters():
            # Extract the index x from the layer name
            x = int(d_name.split('.')[0])
            if x < freeze_up_to_index:
                d_param.requires_grad = False

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
        
        fake_img, _, _ = self.generator(rand1, rand2)
        fake_logit, _, _ = self.discriminator(fake_img, False)
        
        if epoch % 2 == 1:
            image.requires_grad_(True)
            real_logit, _, _ = self.discriminator(image, False)
            d_loss = F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean()
            if epoch % 4 == 1:
                r1_loss = loss.cal_r1_reg(real_logit, image, self.local_rank) * self.args.l_r1
                d_loss += r1_loss
        else:
            real_logit, geometry_feat, appearance_feat = self.discriminator(image, True)
            _, geometry_positive, appearance_negative = self.discriminator(geometry_change, True)
            _, geometry_negative, appearance_positive = self.discriminator(appearance_change, True)

            d_adv_loss = F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean()
            d_aug_loss = (loss.d_contrastive_loss(geometry_feat, geometry_positive, geometry_negative, self.args.tau, self.args.omega) 
                        + loss.d_contrastive_loss(appearance_feat, appearance_positive, appearance_negative, self.args.tau, self.args.omega)) * self.args.l_aux
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
            anchor_image, _, _ = self.generator(rand1, rand2)
            logit, _, _ = self.discriminator(anchor_image, False)
            g_adv_loss = F.softplus(-logit).mean()
            g_loss = g_adv_loss
        else:
            anchor_image, _, _ = self.generator(rand1, rand2)
            resample_geometry, _, _ = self.generator(resample1, rand2)
            resample_appearance, _, _ = self.generator(rand1, resample2)
            
            logit, geometry_feat, appearance_feat = self.discriminator(anchor_image, True)
            _, geometry_positive, appearance_negative = self.discriminator(resample_geometry, True)
            _, geometry_negative, appearance_positive = self.discriminator(resample_appearance, True)

            g_adv_loss = F.softplus(-logit).mean()
            g_aug_loss = (loss.g_contrastive_loss(geometry_feat, geometry_positive, geometry_negative, self.args.tau, self.args.omega) 
                        + loss.g_contrastive_loss(appearance_feat, appearance_positive, appearance_negative, self.args.tau, self.args.omega)) * self.args.l_aux

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

    def feature_visualization(self, psi=1.0, w_psi=0.5):
        # image generation
        rand1 = torch.randn(self.local_batch_size, self.args.geo_noise_dim, device=self.local_rank)
        rand2 = torch.randn(self.local_batch_size, self.args.app_noise_dim, device=self.local_rank)
        
        with torch.no_grad():
            image, feature_maps, flow_maps = self.generator_ema(rand1, rand2, w_psi)    # [-1, 1]
        folder = os.path.join(self.args.model_name, self.args.temp)
        canvas = make_grid(image, nrow=self.local_batch_size)
        canvas = ((canvas + 1) / 2).clamp(0.0, 1.0)
        save_image(canvas, os.path.join(folder, "gen_images.jpg"), padding=0, nrow=1)
        
        num_blocks = int(np.log2(self.args.img_w)) - 2
        for i in range(num_blocks):
            b,c,h,w= feature_maps[i].size()
            features = feature_maps[i]
            for j in range(self.local_batch_size):
                num_images = c
                num_columns = int(num_images ** 0.5)
                feature_images = features[j,0:c,:,:].view(c,1,h,w).to(dtype=torch.float32)
                save_image(make_grid(feature_images, nrow=num_columns, normalize=True), 
                           os.path.join(folder, "{image_num:02d}_feature_{scale_num:02d}_scale.jpg".format(image_num=j, scale_num=i)),
                           padding=0, 
                           nrow=1)

        for i in range(num_blocks):
            b,c,h,w = flow_maps[i].size()
            flowfields = flow_maps[i]
            for j in range(self.local_batch_size):
                new_channel = torch.zeros((1,1,h,w), dtype=torch.float32, device=flowfields.device)
                flow_images = flowfields[j,0:2,:,:].view(1,2,h,w).to(dtype=torch.float32)
                flow_images = torch.cat([flow_images,new_channel],1)
                save_image(make_grid(flow_images, nrow=1, normalize=True), 
                           os.path.join(folder, "{image_num:02d}_flow_{scale_num:02d}_scale.jpg".format(image_num=j, scale_num=i)),
                           padding=0, 
                           nrow=1)


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
                            geometry_change, _, _ = self.generator_ema(
                                inter_code[k*self.local_batch_size:(k+1)*self.local_batch_size,:],
                                appearance_code[k*self.local_batch_size:(k+1)*self.local_batch_size,:],
                                w_psi
                            )
                        canvas.append(geometry_change)
                    canvas = torch.cat(canvas, dim=0)
                    canvas = make_grid(canvas, nrow=nrow, padding=0)
                    canvas = ((canvas + 1) / 2).clamp(0.0, 1.0)
                    # canvas = canvas.clamp(0.0, 1.0)
                    canvas = resize(canvas, size=(disp_resolution * images_per_output // nrow, disp_resolution * nrow), interpolation=InterpolationMode.BILINEAR)
                    frames.append(to_pil(canvas))

                for j in range(num_explore):
                    canvas = []
                    inter_code = geometry_end.lerp(geometry_start, 1/(num_explore)*j)
                    for k in range(images_per_output//self.local_batch_size):
                        with torch.no_grad():
                            geometry_change, _, _ = self.generator_ema(
                                inter_code[k*self.local_batch_size:(k+1)*self.local_batch_size,:],
                                appearance_code[k*self.local_batch_size:(k+1)*self.local_batch_size,:],
                                w_psi
                            )
                        canvas.append(geometry_change)
                    canvas = torch.cat(canvas, dim=0)
                    canvas = make_grid(canvas, nrow=nrow, padding=0)
                    canvas = ((canvas + 1) / 2).clamp(0.0, 1.0)
                    # canvas = canvas.clamp(0.0, 1.0)
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
                            appearance_change, _, _ = self.generator_ema(
                                geometry_code[k*self.local_batch_size:(k+1)*self.local_batch_size,:],
                                inter_code[k*self.local_batch_size:(k+1)*self.local_batch_size,:],
                                w_psi
                            )
                        canvas.append(appearance_change)
                    canvas = torch.cat(canvas, dim=0)
                    canvas = make_grid(canvas, nrow=nrow, padding=0)
                    # canvas = canvas.clamp(0.0, 1.0)
                    canvas = ((canvas + 1) / 2).clamp(0.0, 1.0)
                    canvas = resize(canvas, size=(disp_resolution * images_per_output // nrow, disp_resolution * nrow), interpolation=InterpolationMode.BILINEAR)
                    frames.append(to_pil(canvas))

                for j in range(num_explore):
                    canvas = []
                    inter_code = appearance_end.lerp(appearance_start, 1/(num_explore)*j)
                    for k in range(images_per_output//self.local_batch_size):
                        with torch.no_grad():
                            appearance_change, _, _ = self.generator_ema(
                                geometry_code[k*self.local_batch_size:(k+1)*self.local_batch_size,:],
                                inter_code[k*self.local_batch_size:(k+1)*self.local_batch_size,:],
                                w_psi
                            )
                        canvas.append(appearance_change)
                    canvas = torch.cat(canvas, dim=0)                 
                    canvas = make_grid(canvas, nrow=nrow, padding=0)
                    # canvas = canvas.clamp(0.0, 1.0)
                    canvas = ((canvas + 1) / 2).clamp(0.0, 1.0)
                    canvas = resize(canvas, size=(disp_resolution * images_per_output // nrow, disp_resolution * nrow), interpolation=InterpolationMode.BILINEAR)
                    frames.append(to_pil(canvas))
                    
                # repeat frames 2 times 
                mult_frames.extend(frames * 2)                    
            save_name = os.path.join(self.args.model_name, "samples/appearance_{num}_{b}.mp4".format(num=epoch,b=i))
            self.save_mp4_video(mult_frames, save_name, fps=15)
            
    def save_mp4_video(self, frames, save_path, fps):
        # Get the height and width from the first frame
        width, height = frames[0].size

        # Create a PyAV container with the output filename and format
        output = av.open(save_path, 'w')

        # Create a video stream with H.264 codec
        stream = output.add_stream('libx264', rate=fps)
        stream.width = width
        stream.height = height

        # Open the stream for writing
        stream.open()

        # Iterate over the frames and encode them
        for frame in frames:
            # Convert the PIL image to a numpy array
            frame_np = np.array(frame)

            # Create a PyAV video frame
            video_frame = av.VideoFrame.from_ndarray(frame_np, format='rgb24')

            # Encode the video frame and write it to the output container
            packet = stream.encode(video_frame)

            # Write the encoded packet to the output container
            output.mux(packet)

        # Flush the stream to ensure all frames are written
        output.mux(stream.encode())
        
        # Close the stream and the output container
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
                # image = image * 2.0 - 1.0   # [0,1] to [-1,1]
                feature = inception(image)[0].view(image.shape[0], -1)
                training_features.append(feature.to("cpu"))
        
        gen_features = []
        for i in tqdm(range(num_batches), disable=self.local_rank != 0):
            geometry_code = torch.randn(self.local_batch_size, self.args.geo_noise_dim, device=self.local_rank)
            appearance_code = torch.randn(self.local_batch_size, self.args.app_noise_dim, device=self.local_rank)
            with torch.no_grad():
                fake_images, _, _ = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                # fake_images = fake_images * 2.0 - 1.0   # [0,1] to [-1,1]
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
                fake_images, _, _ = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                
            folder_path = os.path.join(self.args.model_name, 'fakes')
            save_path = os.path.join(folder_path, "{num:04d}_images.jpg".format(num=count))
            save_image(fake_images, save_path, padding=0, nrow=1)
            count = count + 1
            # for b in range(self.local_batch_size):
            #     save_path = os.path.join(folder_path, "{num:04d}_images.jpg".format(num=count))
            #     save_image(fake_images, save_path, padding=0, nrow=1)
            #     count = count + 1

    def check_folder(self, test_dir):
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
                
    def geometry_shared_generation(self, ctrl_dim=-1, num_pairs=100):
        folder_path = os.path.join(self.args.model_name, 'geometry_shared')
        self.check_folder(os.path.join(self.args.model_name, 'geometry_shared'))
        self.check_folder(os.path.join(self.args.model_name, 'geometry_shared/image_a'))
        self.check_folder(os.path.join(self.args.model_name, 'geometry_shared/image_b'))
        count = 0
        if ctrl_dim > -1:
            for ns in tqdm(range(num_pairs), disable=self.local_rank != 0):
                geometry_code = torch.randn(self.local_batch_size, self.args.geo_noise_dim, device=self.local_rank)
                appearance_code = torch.randn(self.local_batch_size, self.args.app_noise_dim, device=self.local_rank)
                appearance_code[:,ctrl_dim:ctrl_dim+1] =-self.args.psi
                with torch.no_grad():
                    image_a, _, _ = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                
                appearance_code[:,ctrl_dim:ctrl_dim+1] = self.args.psi
                with torch.no_grad():
                    image_b, _, _ = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                    
                for b in range(self.local_batch_size):
                    save_path = os.path.join(folder_path, "image_a/{num:04d}_images.jpg".format(num=count))
                    save_image(image_a[b], save_path, padding=0, nrow=1)
                    save_path = os.path.join(folder_path, "image_b/{num:04d}_images.jpg".format(num=count))
                    save_image(image_b[b], save_path, padding=0, nrow=1)
                    count = count + 1
        else:
            for ns in tqdm(range(num_pairs), disable=self.local_rank != 0):
                geometry_code = torch.randn(self.local_batch_size, self.args.geo_noise_dim, device=self.local_rank)
                appearance_code = torch.randn(self.local_batch_size, self.args.app_noise_dim, device=self.local_rank)
                with torch.no_grad():
                    image_a, _, _ = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                
                appearance_code = torch.randn(self.local_batch_size, self.args.app_noise_dim, device=self.local_rank)
                with torch.no_grad():
                    image_b, _, _ = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                    
                for b in range(self.local_batch_size):
                    save_path = os.path.join(folder_path, "image_a/{num:04d}_images.jpg".format(num=count))
                    save_image(image_a[b], save_path, padding=0, nrow=1)
                    save_path = os.path.join(folder_path, "image_b/{num:04d}_images.jpg".format(num=count))
                    save_image(image_b[b], save_path, padding=0, nrow=1)
                    count = count + 1
                                               
    def appearance_shared_generation(self, ctrl_dim=-1, num_pairs=100):
        folder_path = os.path.join(self.args.model_name, 'appearance_shared')
        self.check_folder(os.path.join(self.args.model_name, 'appearance_shared'))
        self.check_folder(os.path.join(self.args.model_name, 'appearance_shared/image_a'))
        self.check_folder(os.path.join(self.args.model_name, 'appearance_shared/image_b'))
        count = 0
        if ctrl_dim > -1:
            for ns in tqdm(range(num_pairs), disable=self.local_rank != 0):
                geometry_code = torch.randn(self.local_batch_size, self.args.geo_noise_dim, device=self.local_rank)
                appearance_code = torch.randn(self.local_batch_size, self.args.app_noise_dim, device=self.local_rank)
                geometry_code[:,ctrl_dim:ctrl_dim+1] =-self.args.psi
                with torch.no_grad():
                    image_a, _, _ = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                
                geometry_code[:,ctrl_dim:ctrl_dim+1] = self.args.psi
                with torch.no_grad():
                    image_b, _, _ = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                    
                for b in range(self.local_batch_size):
                    save_path = os.path.join(folder_path, "image_a/{num:04d}_images.jpg".format(num=count))
                    save_image(image_a[b], save_path, padding=0, nrow=1)
                    save_path = os.path.join(folder_path, "image_b/{num:04d}_images.jpg".format(num=count))
                    save_image(image_b[b], save_path, padding=0, nrow=1)
                    count = count + 1            
        else:
            for ns in tqdm(range(num_pairs), disable=self.local_rank != 0):
                geometry_code = torch.randn(self.local_batch_size, self.args.geo_noise_dim, device=self.local_rank)
                appearance_code = torch.randn(self.local_batch_size, self.args.app_noise_dim, device=self.local_rank)
                with torch.no_grad():
                    image_a, _, _ = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                
                geometry_code = torch.randn(self.local_batch_size, self.args.geo_noise_dim, device=self.local_rank)
                with torch.no_grad():
                    image_b, _, _ = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                    
                for b in range(self.local_batch_size):
                    save_path = os.path.join(folder_path, "image_a/{num:04d}_images.jpg".format(num=count))
                    save_image(image_a[b], save_path, padding=0, nrow=1)
                    save_path = os.path.join(folder_path, "image_b/{num:04d}_images.jpg".format(num=count))
                    save_image(image_b[b], save_path, padding=0, nrow=1)
                    count = count + 1
                
    def joint_interpolation(self, dim1=0, dim2=1, dim3=2, dim4=3, num_images=1):
        interval = self.args.psi * 2 / (self.local_batch_size - 1)
        folder_path = os.path.join(self.args.model_name, 'joint_interpolation')
        self.check_folder(os.path.join(self.args.model_name, 'joint_interpolation'))

        for ns in tqdm(range(num_images), disable=self.local_rank != 0):
            sub_folder = os.path.join(folder_path, "{num:03d}".format(num=ns))
            self.check_folder(sub_folder)
            
            init_code = torch.randn(1, self.args.geo_noise_dim + self.args.app_noise_dim, device=self.local_rank)
            # interpolation according to dim1 and dim2
            for i in range(self.local_batch_size):
                code = init_code.repeat([self.local_batch_size,1])
                code[:, dim1] = -self.args.psi + interval*i
                code[:, dim2] = torch.linspace(-self.args.psi, self.args.psi, steps=self.local_batch_size, device=self.local_rank)
                geometry_code, appearance_code = torch.chunk(code, chunks=2, dim=1)
                with torch.no_grad():
                    image_a, _, _ = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                
                for j in range(self.local_batch_size):
                    save_path = os.path.join(sub_folder, "d1_d2_images({i:01d},{j:01d}).jpg".format(num=ns, i=i, j=j))
                    save_image(image_a[j], save_path, padding=0, nrow=1)
                
            # interpolation according to dim2 and dim3
            for i in range(self.local_batch_size):
                code = init_code.repeat([self.local_batch_size,1])
                code[:, dim2] = -self.args.psi + interval*i
                code[:, dim3] = torch.linspace(-self.args.psi, self.args.psi, steps=self.local_batch_size, device=self.local_rank)
                geometry_code, appearance_code = torch.chunk(code, chunks=2, dim=1)
                with torch.no_grad():
                    image_a, _, _ = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                    
                for j in range(self.local_batch_size):
                    save_path = os.path.join(sub_folder, "d2_d3_images({i:01d},{j:01d}).jpg".format(num=ns, i=i, j=j))
                    save_image(image_a[j], save_path, padding=0, nrow=1)
                            
            # interpolation according to dim3 and dim4
            for i in range(self.local_batch_size):
                code = init_code.repeat([self.local_batch_size,1])
                code[:, dim3] = -self.args.psi + interval*i
                code[:, dim4] = torch.linspace(-self.args.psi, self.args.psi, steps=self.local_batch_size, device=self.local_rank)
                geometry_code, appearance_code = torch.chunk(code, chunks=2, dim=1)
                with torch.no_grad():
                    image_a, _, _ = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                    
                for j in range(self.local_batch_size):
                    save_path = os.path.join(sub_folder, "d3_d4_images({i:01d},{j:01d}).jpg".format(num=ns, i=i, j=j))
                    save_image(image_a[j], save_path, padding=0, nrow=1)
                                    
            # interpolation according to dim4 and dim1
            for i in range(self.local_batch_size):
                code = init_code.repeat([self.local_batch_size,1])
                code[:, dim4] = -self.args.psi + interval*i
                code[:, dim1] = torch.linspace(-self.args.psi, self.args.psi, steps=self.local_batch_size, device=self.local_rank)
                geometry_code, appearance_code = torch.chunk(code, chunks=2, dim=1)
                with torch.no_grad():
                    image_a, _, _ = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                    
                for j in range(self.local_batch_size):
                    save_path = os.path.join(sub_folder, "d4_d1_images({i:01d},{j:01d}).jpg".format(num=ns, i=i, j=j))
                    save_image(image_a[j], save_path, padding=0, nrow=1)
                    
                    
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
                    image, _, _ = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                                
                n_rows = int(self.local_batch_size ** 0.5)
                canvas = make_grid(image, nrow=n_rows, padding=0)
                canvas = canvas.clamp(0.0, 1.0)
                frames.append(to_pil(canvas))
                
            for i in range(num_explore):
                latent_code[:,controlled_dim] = latent_code[:,controlled_dim] - interval
                geometry_code, appearance_code = torch.chunk(latent_code, chunks=2, dim=1)
                with torch.no_grad():
                    image, _, _ = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                                
                n_rows = int(self.local_batch_size ** 0.5)
                canvas = make_grid(image, nrow=n_rows, padding=0)
                canvas = canvas.clamp(0.0, 1.0)
                frames.append(to_pil(canvas))

            # repeat frames 2 times 
            mult_frames.extend(frames * num_repeat)
            save_name = os.path.join(self.args.model_name, "demo/controlled_dim={num}_{n}.mp4".format(num=controlled_dim,n=n))
            self.save_mp4_video(mult_frames, save_name, fps=num_explore)
        
        
    def latent_exploration(self, num_explore=5, num_images=1):
        folder_path = os.path.join(self.args.model_name, 'latent_exploration')
        self.check_folder(folder_path)
        
        count = 0
        for i in range(num_images):
            for j in range(self.args.geo_noise_dim + self.args.app_noise_dim):
                latent_code = torch.randn(self.local_batch_size, self.args.geo_noise_dim + self.args.app_noise_dim, device=self.local_rank)
                
                latent_code[:,j] = -self.args.psi
                geometry_code, appearance_code = torch.chunk(latent_code, chunks=2, dim=1)
                with torch.no_grad():
                    canvas, _, _ = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)

                interval = self.args.psi*2.0/(num_explore-1)
                for k in range(num_explore-1):
                    latent_code[:,j] = latent_code[:,j] + interval
                    geometry_code, appearance_code = torch.chunk(latent_code, chunks=2, dim=1)
                    with torch.no_grad():
                        image, _, _ = self.generator_ema(geometry_code, appearance_code, self.args.w_psi)
                    canvas = torch.concat((canvas, image), dim=0)
                
                for b in range(self.local_batch_size):
                    selected_images = canvas[b::self.local_batch_size]
                    selected_images = ((selected_images + 1) / 2).clamp(0.0, 1.0)
                    save_name = os.path.join(folder_path, "{dim:04d}_{b:04d}_{n:04d}_images.jpg".format(dim=j,b=b,n=i))
                    save_image(selected_images, save_name, padding=0, nrow=num_explore)
                    count = count + 1