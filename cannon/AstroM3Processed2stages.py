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
from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore
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
                    "time": self.dataset[idx]['photometry'][:, 0]
                    }
        #breakpoint()
        return {"spectra": res, "photometry": photores, 
                #"metadata": {"metadta": self.dataset[idx]['metadata'], 
                #             "label": self.dataset[idx]['label']}
                }
        

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", 
                            #init_method=f"file://./tmp/ddp_init",
                            init_method="tcp://127.0.0.1:23456",
                            rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()


'''
def train(epoch=1000, lr = 2.5e-4, bottlenecklen = 16, bottleneckdim = 16, 
          concat = True, 
          spectra_tokens = 128,
          photometry_tokens = 128,
          model_dim = 128, encoder_layers = 4, 
          decoder_layers = 4,regularize = 0.000, 
          dropping_prob = 0.3,
          batch = 4, aug = 3, save_every = 20):
    
    

    training_data = AstroM3Procesed(aug=aug)
    

    training_loader = DataLoader(training_data, batch_size = batch, 
                                 collate_fn = padding_collate_fun(supply = ['flux', 'wavelength', 'time'], 
                                                                  mask_by = "flux", 
                                                                  multimodal = True), 
                                 shuffle = True, num_workers=4, pin_memory=True)
    
    #breakpoint()
    tokenizers = {
        "spectra": spectraTransceiverEncoder(
            bottleneck_length = spectra_tokens,
            bottleneck_dim = model_dim,
            model_dim = model_dim    
        ), 
        "photometry": photometricTransceiverEncoder(
            
            num_bands = 1, 
            bottleneck_length = photometry_tokens,
            bottleneck_dim = model_dim,
            model_dim = model_dim, 
        )
    }
    
    encoder = PerceiverEncoder(
                    bottleneck_length = bottlenecklen,
                    bottleneck_dim = bottleneckdim,
                    model_dim = model_dim,
                    num_layers = encoder_layers,
                    ff_dim = model_dim,
                    num_heads = 4 
    )
    
    
    scores = {
        "spectra":spectraTransceiverScore(
                    bottleneck_dim = bottleneckdim,
                    model_dim = model_dim,
                    num_layers = decoder_layers,
                    concat = concat
                    ), 
        "photometry": photometricTransceiverScore(
            bottleneck_dim = bottleneckdim,
                 num_bands = 1,
                 model_dim = model_dim,
                 num_layers = decoder_layers,
                 concat = concat
        )
    }

    

    mydaep = multimodaldaep(tokenizers, encoder, scores, 
                            measurement_names = {"spectra":"flux", "photometry": "flux"}, 
                            modality_dropping_during_training = partial(modality_drop, p_drop=dropping_prob)).to(device)
    
    mydaep.train()
    optimizer = AdamW(mydaep.parameters(), lr=lr)
    epoch_loss = []
    epoches = []
    target_save = None
    progress_bar = tqdm(range(epoch))
    for ep in progress_bar:
        losses = []
        for x in tqdm(training_loader):
            x = to_device(x)
            #breakpoint()
            #print(x['spectra']['flux'].min().item())
            optimizer.zero_grad()
            #with torch.cuda.amp.autocast():
            loss = mydaep(x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        this_epoch = np.array(losses).mean().item()
        epoch_loss.append(math.log(this_epoch))
        epoches.append(ep)
        if (ep+1) % save_every == 0:
            if target_save is not None:
                os.remove(target_save)
            target_save = f"../ckpt/AstroM3_daep2stages_{bottlenecklen}-{bottleneckdim}-{spectra_tokens}-{photometry_tokens}-{encoder_layers}-{decoder_layers}-{model_dim}_concat{concat}_lr{lr}_modaldropP{dropping_prob}_epoch{ep+1}_batch{batch}_reg{regularize}_aug{aug}.pth"
            torch.save(mydaep, target_save)
            plt.plot(epoches, epoch_loss)
            plt.show()
            plt.savefig(f"./logs/AstroM3_daep2stages_{bottlenecklen}-{bottleneckdim}-{spectra_tokens}-{photometry_tokens}-{encoder_layers}-{decoder_layers}-{model_dim}_concat{concat}_lr{lr}_modaldropP{dropping_prob}_batch{batch}_reg{regularize}_aug{aug}.png")
            plt.close()
        progress_bar.set_postfix(loss=f"epochs:{ep}, {math.log(this_epoch):.4f}") 
    
'''


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
        num_heads=4
    )

    scores = {
        "spectra": spectraTransceiverScore(
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
    progress_bar = range(args["epoch"])

    for ep in progress_bar:
        model.train()
        sampler.set_epoch(ep)
        epoch_losses = []

        for batch in dataloader: #tqdm(dataloader, disable=rank != 0):
            batch = to_device(batch, device)
            optimizer.zero_grad()

            with autocast('cuda'):
                loss = model(batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(loss.item())

        if rank == 0:
            avg_loss = np.mean(epoch_losses)
            losses_log.append(math.log(avg_loss))
            print(f"[GPU {rank}] Epoch {ep} log-loss: {math.log(avg_loss):.4f}")
            if (ep + 1) % args["save_every"] == 0:
                ckpt_name = f"../ckpt/AstroM3_daep2stages_ddp_{args['bottlenecklen']}-{args['bottleneckdim']}-{args['spectra_tokens']}-{args['photometry_tokens']}-{args['encoder_layers']}-{args['decoder_layers']}-{args['model_dim']}_concat{args['concat']}_crossattnonly{args['cross_attn_only']}_lr{args['lr']}_modaldropP{args['dropping_prob']}_epoch{ep+1}_batch{args['batch']}_world{world_size}_reg0.0_aug{args['aug']}.pth"
                torch.save(model, ckpt_name)
                plt.plot(losses_log)
                plt.savefig(ckpt_name.replace("pth", "loss.png"))
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