from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset,DistributedSampler
import torch


import torch.multiprocessing as mp
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
#from torch.cuda.amp import autocast
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial


import numpy as np
# optimizer
from torch.optim import AdamW
from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'


##### daep #####
from daep.data_util import to_tensor, collate_fn_stack, to_device, padding_collate_fun
from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore2stages
from daep.daep import multimodaldaep, modality_drop
from daep.tokenizers import photometryTokenizer, spectraTokenizer
from daep.Perceiver import PerceiverEncoder
from daep.PhotometricLayers import photometricTransceiverEncoder, photometricTransceiverScore
from functools import partial

import math 
import os
from tqdm import tqdm



class AstroM3ProcessedPreAug(Dataset):
    def __init__(self, name="full_42", which="train", aug=1):
        assert aug >= 1 and isinstance(aug, int)
        dataset = load_dataset("AstroMLCore/AstroM3Processed", name=name)[which]
        dataset.set_format(type="torch")

        self.data = []
        print("Loading and augmenting data...")
        for i in tqdm(range(len(dataset))):
            for j in range(aug):
                spectra_flux = dataset[i]['spectra'][1]
                spectra_wave = dataset[i]['spectra'][0]
                spectra_noise = dataset[i]['spectra'][2]

                flux = torch.log10(
                    spectra_flux + (
                        (torch.randn_like(spectra_flux) * spectra_noise) if aug > 1 else 0
                    )
                ) + 4.782

                phot = dataset[i]['photometry']
                phot_flux = phot[:, 1] + (
                    (torch.randn_like(phot[:, 0]) * phot[:, 2]) if aug > 1 else 0
                )

                self.data.append({
                    "spectra": {
                        "flux": flux,
                        "wavelength": spectra_wave,
                        "phase": torch.tensor(0.)
                    },
                    "photometry": {
                        "flux": phot_flux,
                        "time": phot[:, 0]
                    }
                })
        print("Done!")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class AstroM3Dataset(Dataset):
    def __init__(self, name = "full_42", which = "train", aug = None):
        # Load the default full dataset with seed 42
        assert aug is None or (aug >=1 and isinstance(aug, int)), "Augmentation has to be positive integer >=1 or None for not augmenting"
        self.dataset = load_dataset("../../AstroM3Dataset", name=name, trust_remote_code=True)[which]
        self.dataset.set_format(type="torch")
        self.aug = aug if aug is not None else 1
        self.which = which
        if which == "test" and self.aug > 1:
            print("We do not augment test")
            self.aug = 1
        #breakpoint()

    def __len__(self):
        return self.aug * len(self.dataset)

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        
        res = {"flux": (torch.log10(self.dataset[idx]['spectra'][:, 1] + (
                        (torch.randn_like(self.dataset[idx]['spectra'][:, 2]) * \
                        self.dataset[idx]['spectra'][:, 2]) if self.aug > 1 else 0. ) ) - 2.8766)/0.7795  # this is the mean
                        , 
               "wavelength": (self.dataset[idx]['spectra'][:, 0] - 6000.1543)/1548.8627, 
               "phase": torch.tensor(0.)}       
        
        #breakpoint()
        return res


class AstroM3Dataset(Dataset):
    def __init__(self, name = "full_42", which = "train", aug = None):
        # Load the default full dataset with seed 42
        assert aug is None or (aug >=1 and isinstance(aug, int)), "Augmentation has to be positive integer >=1 or None for not augmenting"
        self.dataset = load_dataset("../../AstroM3Dataset", name=name, trust_remote_code=True)[which]
        self.dataset.set_format(type="torch")
        self.aug = aug if aug is not None else 1
        self.which = which
        if which == "test" and self.aug > 1:
            print("We do not augment test")
            self.aug = 1
        #breakpoint()
        for i in len(self.dataset):
            self.dataset.dataset[i]['photometry'][:, 0] -= self.dataset.dataset[i]['photometry'][:, 0].min()

    def __len__(self):
        return self.aug * len(self.dataset)

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        
        res = {"flux": (torch.log10(self.dataset[idx]['spectra'][:, 1] + (
                        (torch.randn_like(self.dataset[idx]['spectra'][:, 2]) * \
                        self.dataset[idx]['spectra'][:, 2]) if self.aug > 1 else 0. ) ) - 2.8766)/0.7795  # this is the mean
                        , 
               "wavelength": (self.dataset[idx]['spectra'][:, 0] - 6000.1543)/1548.8627, 
               "phase": torch.tensor(0.)}      
        
        
        photores = {
            "flux": (self.dataset[idx]['photometry'][:, 1] + (
                        (torch.randn_like(self.dataset[idx]['photometry'][:, 0]) * \
                        self.dataset[idx]['photometry'][:, 2]) if self.aug > 1 else 0.) - 22.6879)/27.7245 , # noise added only if augmentation and training
            "time": (self.dataset[idx]['photometry'][:, 0]- 788.0814)/475.3434 # [-3, 3] kinda aribitrary to match standardized range
        } 
        
        #breakpoint()
        return {"spectra": res, "photometry": photores}


'''
class AstroM3Processed(Dataset):
    def __init__(self, name = "full_42", which = "train", aug = None):
        # Load the default full dataset with seed 42
        assert aug is None or (aug >=1 and isinstance(aug, int)), "Augmentation has to be positive integer >=1 or None for not augmenting"
        self.dataset = load_dataset("AstroMLCore/AstroM3Processed", name=name)[which]
        self.dataset.set_format(type="torch")
        self.aug = aug if aug is not None else 1
        self.which = which
        if which == "test" and aug > 1:
            print("We do not augment test")
            self.aug = 1
        #breakpoint()

    def __len__(self):
        return self.aug * len(self.dataset)

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        
        res = {"flux": torch.log10( self.dataset[idx]['spectra'][1] + (
                        (torch.randn_like(self.dataset[idx]['spectra'][0]) * \
                        self.dataset[idx]['spectra'][2]) if self.aug > 1 else 0. )) + 4.782 # this is the mean
                        , 
               "wavelength": self.dataset[idx]['spectra'][0], 
               "phase": torch.tensor(0.)}       
        
        photores = {"flux": self.dataset[idx]['photometry'][:, 1] + (
                        (torch.randn_like(self.dataset[idx]['photometry'][:, 0]) * \
                        self.dataset[idx]['photometry'][:, 2]) if self.aug > 1 else 0.) , # noise added only if augmentation and training
                    "time": self.dataset[idx]['photometry'][:, 0] * 6 - 3 # [-3, 3] kinda aribitrary to match standardized range
                    }
        #breakpoint()
        return {"spectra": res, "photometry": photores, 
                #"metadata": {"metadta": self.dataset[idx]['metadata'], 
                #             "label": self.dataset[idx]['label']}
                }
'''       

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", 
                            #init_method=f"file://./tmp/ddp_init",
                            init_method="tcp://127.0.0.1:23456",
                            rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def train_worker(rank, world_size, args):
    setup_ddp(rank, world_size)

    # Dataset and loader with distributed sampler
    dataset = AstroM3Processed(aug=args["aug"])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args["batch"], sampler=sampler,
                            collate_fn=padding_collate_fun(supply=['flux', 'wavelength', 'time'],
                                                           mask_by="flux", multimodal=True),
                            num_workers=1, pin_memory=True)

    device = torch.device(f"cuda:{rank}")
    
    # Build model
    tokenizers = {
        "spectra": spectraTransceiverEncoder(
            bottleneck_length=args["spectra_tokens"],
            bottleneck_dim=args["model_dim"],
            model_dim=args["model_dim"]
        ),
        "photometry": photometricTransceiverEncoder(
            num_bands=1,
            bottleneck_length=args["photometry_tokens"],
            bottleneck_dim=args["model_dim"],
            model_dim=args["model_dim"]
        )
    }

    encoder = PerceiverEncoder(
        bottleneck_length=args["bottlenecklen"],
        bottleneck_dim=args["bottleneckdim"],
        model_dim=args["model_dim"],
        num_layers=args["encoder_layers"],
        ff_dim=args["model_dim"],
        num_heads=4,
        self_attn = args['mixerselfattn']
    )

    scores = {
        "spectra": spectraTransceiverScore2stages(
            bottleneck_dim=args["bottleneckdim"],
            model_dim=args["model_dim"],
            num_layers=args["decoder_layers"],
            concat=args["concat"],
            cross_attn_only=args['cross_attn_only']
        ),
        "photometry": photometricTransceiverScore(
            bottleneck_dim=args["bottleneckdim"],
            num_bands=1,
            model_dim=args["model_dim"],
            num_layers=args["decoder_layers"],
            concat=args["concat"],
            cross_attn_only=args['cross_attn_only']
        )
    }

    model = multimodaldaep(
        tokenizers, encoder, scores,
        measurement_names={"spectra": "flux", "photometry": "flux"},
        modality_dropping_during_training=partial(modality_drop, p_drop=args["dropping_prob"])
    ).to(device)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True) # modality dropping make it tricky to avoid unused parameters
    optimizer = AdamW(model.parameters(), lr=args["lr"])
    scaler = GradScaler('cuda')

    losses_log = []
    progress_bar = tqdm(range(args["epoch"]), disable=rank != 0)
    ckpt_name = None
    for ep in progress_bar:
        model.train()
        sampler.set_epoch(ep)
        epoch_losses = []

        for batch in dataloader: #tqdm(dataloader, disable=rank != 0):
            batch = to_device(batch, device)
            optimizer.zero_grad()

            with autocast('cuda'):
                loss = model(batch)
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ NaN or Inf loss detected — skipping step")
                continue  # skip this step and avoid corrupting gradients

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(loss.item())
        
        if rank == 0:
            avg_loss = np.mean(epoch_losses)
            losses_log.append(math.log(avg_loss))
            print(f"[GPU {rank}] Epoch {ep} log-loss: {math.log(avg_loss):.4f}")
            if (ep + 1) % args["save_every"] == 0:
                if ckpt_name is not None:
                    os.remove(ckpt_name)
                ckpt_name = f"../ckpt/AstroM3_daep2stages_ddp_{args['bottlenecklen']}-{args['bottleneckdim']}-{args['spectra_tokens']}-{args['photometry_tokens']}-{args['encoder_layers']}-{args['decoder_layers']}-{args['model_dim']}_concat{args['concat']}_crossattnonly{args['cross_attn_only']}_lr{args['lr']}_modaldropP{args['dropping_prob']}_epoch{ep+1}_batch{args['batch']}_world{world_size}_reg0.0_aug{args['aug']}.pth"
                torch.save(model.module.state_dict(), ckpt_name)
                plt.plot(losses_log)
                plt.savefig(ckpt_name.replace("pth", "loss.png").replace("../ckpt", "./logs").replace(f"_epoch{ep+1}", ""))
                plt.close()

    cleanup_ddp()




import fire           

def main(**kwargs):
    world_size = torch.cuda.device_count()
    print(f"training with world size {world_size}")
    mp.spawn(train_worker,
             args=(world_size, kwargs),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    fire.Fire(main)