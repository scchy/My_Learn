# python3
# Create Date: 2023-11-14
# Func: distribute training
# =============================================================================

import torch 
from torch import optim
from torch.nn import functional as F
from torch import nn
import datasets
import transformers
from datasets import load_dataset
from accelerate import Accelerator
from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter
import logging
import wandb
from torch.utils.data.dataloader import DataLoader
# from trl.trainer import ConstantLengthDataset
from transformers import pipeline, set_seed 

# device = 'cpu'
accelerator = Accelerator()
# model = nn.Transformer().to(device)
model = nn.Transformer()
opt = optim.Adam(model.parameters())
data_set = load_dataset('my_dataset')
data = torch.utils.data.DataLoader(data_set, shuffle=True)
model, opt, data = accelerator.prepare(model, opt, data)

model.train()
for epoch in range(10):
    for x, target in data:
        # x, target = x.to(device), target.to(device)
        opt.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, target)
        # loss.backward()
        accelerator.backward()
        opt.step()
    

config = {
    "train_batch_size": 2, # 12
    "valid_batch_size": 2, # 12
    "weight_decay": 0.1,
    "shuffle_buffer": 1000,
    "learning_rate": 2e-4, # 5e-4
    "lr_scheduler_type": "cosine",
    "num_warmup_steps": 750, # 2000
    "gradient_accumulation_steps": 16, # 1
    "max_train_steps": 50000, # 150000
    "max_eval_steps": -1,
    "seq_length": 1024,
    "seed": 1,
    "save_checkpoint_steps": 50000
}
args = Namespace(**config)

# 2- logging
def setup_logging(project_name):
    """
    Each worker gets a unique accelerator.process_index, 
    which we use with the File Handler to write the logs of each worker to an individual file.
    
    run_anme: which we use later to name our experiment branch on the Hub.
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctimes)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"log/debug_{accelerator.process_index}.log"),
            logging.StreamHandler()
        ]
    )
    tb_writer = None
    if accelerator.is_main_process:
        wandb.init(
            project=project_name,
            config=args
        )
        run_name = wandb.run.name
        # tb_writer = SummaryWriter()
        # tb_writer.add_hparams(vars(args), {'0': 0})
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_debug()
        transformers.utils.logging.set_verbosity_info()
    else:
        run_name = ''
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger, tb_writer, run_name


# log metrics
def log_metrics(logger, step, metrics):
    logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        wandb.log(metrics)
        #[tb_writer.add_scalar(k, v, step) for k, v in metrics.items()]


# 3- dataLoaders
def create_dataloaders(tokenizer, dataset_name):
    tr_data = load_dataset(dataset_name + '-train', split='train', streaming=True)
    tr_data = tr_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    val_data = load_dataset(dataset_name + '-vaild', split='validation', streaming=True)
    
    train_datset = ConstantLengthDataset(
        tokenizer, tr_data, seq_length=args.seq_length
    )
    valid_datset = ConstantLengthDataset(
        tokenizer, val_data, seq_length=args.seq_length
    )
    tr_dataLoader = DataLoader(train_datset, batch_size=args.train_batch_size)
    val_dataLoader = DataLoader(valid_datset, batch_size=args.train_batch_size)
    return tr_dataLoader, val_dataLoader


# biases and layerNorm weights are not subject to weight decay
def get_grouped_params(model, no_decay=['bias', 'LayerNorm.wight']):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {'params': params_with_wd, 'weight_decay': args.weight_decay},
        {'params': params_without_wd, 'weight_decay': 0.0},
    ]


# 4- evaluate
def evaluate():
    """_summary_
    The perplexity measures how well the models output probabilty
    distributions predict the targeted tokens.
    So a lower perplexity by exponentiating the cross-entropy loss 
    which we get from the models output
    """
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if args.max_eval_steps > 0 and step >= args.max_eval_steps:
            break
    
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = torch.tensor(float('inf'))
    return loss.item(), perplexity.item()


