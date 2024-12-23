import sys
sys.path.insert(0, "D:\\Python Code\\daclip-uir-main\\daclip-uir-main\\da-clip\\src\\training")
sys.path.insert(0, "D:\\Python Code\\daclip-uir-main\\daclip-uir-main\\da-clip\\src")
#sys.path.remove("C:\\Users\\Administrator\\AppData\\Roaming\\Python\Python39\\site-packages")

#print(sys.path)
#print(11111111111111)
import glob
import logging
import os
import re
import subprocess

import random
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


#from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from open_clip.factory import create_model_and_transforms,  get_tokenizer, create_loss
from open_clip.model import trace_model
from training.data import get_data
from training.distributed import is_master, init_distributed_device, broadcast_object
from training.logger import setup_logging
from training.params import parse_args
# import params as parse_args


from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from training.train import train_one_epoch, evaluate
from training.file_utils import pt_load, check_exists, start_sync_process, remote_sync


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):              #按数字自然排序
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):

    # print(1, args)
    #
    # print(args.model)
    # print(args.pretrained)
    # print(args.precision)
    # print(args.torchscript)
    # print(args.force_quick_gelu)
    # print(args.force_custom_text)
    # print(args.force_patch_dropout)
    # print(args.force_image_size)
    # print(args.pretrained_image)
    # print(args.image_mean)
    # print(args.image_std)
    # print(args.aug_cfg)





    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:  #no
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    # if is_master(args, local=args.log_local):
    #     os.makedirs(log_base_path, exist_ok=True)
    #     log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
    #     args.log_path = os.path.join(log_base_path, log_filename)
    #     if os.path.exists(args.log_path) and not resume_latest:
    #         print(
    #             "Error. Experiment already exists. Use --name {} to specify a new experiment."
    #         )
    #         return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    print(args.checkpoint_path)
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
    #no
    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from
    #no
    if args.copy_codebase:
        copy_codebase(args)
    #no
    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        #FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        #FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)


    #print(args)
    #print([args.model,
    #      args.pretrained,
    #      args.precision,
    #      device,
    #      args.torchscript,
    #      args.force_quick_gelu,
    #      args.force_custom_text,
    #      args.force_patch_dropout,
    #      args.force_image_size,
    #      args.pretrained_image,
    #      args.image_mean,
    #      args.image_std,
    #      args.aug_cfg,
    #      True,])

    model, preprocess_train, preprocess_val =create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,

    )
    if args.distill:                                   #no
        # FIXME: currenlty assumes the model your distilling from has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model, 
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
        )
    if args.use_bnb_linear is not None:                                                   #no
        print('=> using a layer from bitsandbytes.\n'
              '   this is an experimental feature which requires two extra pip installs\n'
              '   pip install bitsandbytes triton'
              '   please make sure to use triton 2.0.0')
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear
        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)

    random_seed(args.seed, args.rank)

    if args.trace: #NO
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image: #NO
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text: #NO
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing: #NO
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:                 #no
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    
        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None




    if args.train_data or args.dataset_type == "synthetic": #y
        assert not args.trace, 'Cannot train with traced model'

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)
        #######################################################################
        prompt_learner=[]
        for name, param in model.named_parameters():
            if "adapter" in name:
                param.requires_grad_(True)

        # for name, param in model.named_parameters():  # 假设model是你的模型对象
        #     if "prompt_learner" in name :
        #         # 将符合条件的参数存储在prompt_learner中用于优化
        #         prompt_learner.append(param)
        ########################################################################

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        # optimizer = optim.SGD(
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},


            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        if args.horovod:   #no
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:   #no
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    #{'train': DataInfo(dataloader=<torch.utils.data.dataloader.DataLoader object at 0x0000018A3BAEA220>, sampler=None, shared_epoch=None), 'val': DataInfo(dataloader=<torch.utils.data.dataloader.DataLoader object at 0x0000018A22AB73A0>, sampler=None, shared_epoch=None)}

    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=get_tokenizer(args.model))


    #print(args, preprocess_train, preprocess_val)
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:    #y
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":          #y
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):                                          #no
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    if args.torchcompile:                              #no
        logging.info('Compiling model...')
        model = torch.compile(model)

    if 'train' not in data:                                #no
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        # Evaluate.
        evaluate(model, data, start_epoch, args, writer)
        return

    loss = create_loss(args)

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, completed_epoch, args, writer)

        # Saving checkpoints.

        #########################################
        # 定义变量来追踪最佳验证性能和对应的轮次
        best_valid_performance = float('inf')
        best_epoch = 0
        ###########################################


        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()





            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')
    

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


import argparse
import ast


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train-data",
    type=str,
    default="D:/Python Code/daclip-uir-main/daclip-uir-main/datasets/universal/daclip_train.csv",
    help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
)
parser.add_argument(
    "--train-data-upsampling-factors",
    type=str,
    default=None,
    help=(
        "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
        "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
        "By default, datapoints are sampled uniformly regardless of the dataset sizes."
    )
)
parser.add_argument(
    "--val-data",
    type=str,
    default="D:/Python Code/daclip-uir-main/daclip-uir-main/datasets/universal/daclip_val.csv",
    help="Path to file(s) with validation data",
)
parser.add_argument(
    "--train-num-samples",
    type=int,
    default=None,
    help="Number of samples in dataset. Required for webdataset if not available in info file.",
)
parser.add_argument(
    "--val-num-samples",
    type=int,
    default=None,
    help="Number of samples in dataset. Useful for webdataset if not available in info file.",
)
parser.add_argument(
    "--dataset-type",
    choices=["webdataset", "csv", "synthetic", "auto"],
    default="auto",
    help="Which type of dataset to process."
)
parser.add_argument(
    "--dataset-resampled",
    default=False,
    action="store_true",
    help="Whether to use sampling with replacement for webdataset shard selection."
)
parser.add_argument(
    "--csv-separator",
    type=str,
    default="\t",
    help="For csv-like datasets, which separator to use."
)
parser.add_argument(
    "--csv-img-key",
    type=str,
    default="filepath",
    help="For csv-like datasets, the name of the key for the image paths."
)
parser.add_argument(
    "--csv-caption-key",
    type=str,
    default="title",
    help="For csv-like datasets, the name of the key for the captions."
)
parser.add_argument(
    "--imagenet-val",
    type=str,
    default=None,
    help="Path to imagenet val set for conducting zero shot evaluation.",
)
parser.add_argument(
    "--imagenet-v2",
    type=str,
    default=None,
    help="Path to imagenet v2 for conducting zero shot evaluation.",
)
parser.add_argument(
    "--logs",
    type=str,
    default="./logs/",
    help="Where to store tensorboard logs. Use None to avoid storing logs.",
)
parser.add_argument(
    "--log-local",
    action="store_true",
    default=False,
    help="log files on local master, otherwise global master only.",
)
parser.add_argument(
    "--name",
    type=str,
    default="3dataset",
    help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
)
parser.add_argument(
    "--workers", type=int, default=1, help="Number of dataloader workers per GPU."
)
parser.add_argument(
    "--batch-size", type=int, default=64, help="Batch size per GPU."
)
parser.add_argument(
    "--epochs", type=int, default=150, help="Number of epochs to train for."
)
parser.add_argument(
    "--epochs-cooldown", type=int, default=None,
    help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards."
)
parser.add_argument("--lr", type=float, default=0.002, help="Learning rate.")      #2e-5
parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta 2.")
parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon.")
parser.add_argument("--wd", type=float, default=0.05, help="Weight decay.")
parser.add_argument(
    "--warmup", type=int, default=10000, help="Number of steps to warmup for."
)
parser.add_argument(
    "--use-bn-sync",
    default=False,
    action="store_true",
    help="Whether to use batch norm sync.")
parser.add_argument(
    "--skip-scheduler",
    action="store_true",
    default=False,
    help="Use this flag to skip the learning rate decay.",
)
parser.add_argument(
    "--lr-scheduler",
    type=str,
    default='cosine',
    help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
)
parser.add_argument(
    "--lr-cooldown-end", type=float, default=0.0,
    help="End learning rate for cooldown schedule. Default: 0"
)
parser.add_argument(
    "--lr-cooldown-power", type=float, default=1.0,
    help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)"
)
parser.add_argument(
    "--save-frequency", type=int, default=1, help="How often to save checkpoints."
)
parser.add_argument(
    "--save-most-recent",
    action="store_true",
    default=False,
    help="Always save the most recent model trained to epoch_latest.pt.",
)
parser.add_argument(
    "--zeroshot-frequency", type=int, default=1, help="How often to run zero shot."
)
parser.add_argument(
    "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
)
parser.add_argument(
    "--resume",
    default=None,
    type=str,
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--precision",
    choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
    default="amp",
    help="Floating point precision."
)
parser.add_argument(
    "--model",
    type=str,
    default="daclip_ViT-B-32",
    help="Name of the vision backbone to use.",
)
parser.add_argument(
    "--pretrained",
    default='laion2b_s34b_b79k',
    type=str,
    help="Use a pretrained CLIP model weights with the specified tag or file path.",
)
parser.add_argument(
    "--pretrained-image",
    default=False,
    action='store_true',
    help="Load imagenet pretrained weights for image tower backbone if available.",
)
parser.add_argument(
    "--lock-image",
    default=False,
    action='store_true',
    help="Lock full image tower by disabling gradients.",
)
parser.add_argument(
    "--lock-image-unlocked-groups",
    type=int,
    default=0,
    help="Leave last n image tower layer groups unlocked.",
)
parser.add_argument(
    "--lock-image-freeze-bn-stats",
    default=False,
    action='store_true',
    help="Freeze BatchNorm running stats in image tower for any locked layers.",
)
parser.add_argument(
    '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
    help='Override default image mean value of dataset')
parser.add_argument(
    '--image-std', type=float, nargs='+', default=None, metavar='STD',
    help='Override default image std deviation of of dataset')
parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
parser.add_argument(
    "--grad-checkpointing",
    default=False,
    action='store_true',
    help="Enable gradient checkpointing.",
)
parser.add_argument(
    "--local-loss",
    default=False,
    action="store_true",
    help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
)
parser.add_argument(
    "--gather-with-grad",
    default=False,
    action="store_true",
    help="enable full distributed gradient for feature gather"
)
parser.add_argument(
    '--force-image-size', type=int, nargs='+', default=None,
    help='Override default image size'
)
parser.add_argument(
    "--force-quick-gelu",
    default=False,
    action='store_true',
    help="Force use of QuickGELU activation for non-OpenAI transformer models.",
)
parser.add_argument(
    "--force-patch-dropout",
    default=None,
    type=float,
    help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
)
parser.add_argument(
    "--force-custom-text",
    default=False,
    action='store_true',
    help="Force use of CustomTextCLIP model (separate text-tower).",
)
parser.add_argument(
    "--torchscript",
    default=False,
    action='store_true',
    help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
)
parser.add_argument(
    "--torchcompile",
    default=False,
    action='store_true',
    help="torch.compile() the model, requires pytorch 2.0 or later.",
)
parser.add_argument(
    "--trace",
    default=False,
    action='store_true',
    help="torch.jit.trace the model for inference / eval only",
)
parser.add_argument(
    "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
)
# arguments for distributed training
parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--report-to",
    default='tensorboard',
    type=str,
    help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
)
parser.add_argument(
    "--wandb-notes",
    default='',
    type=str,
    help="Notes if logging with wandb"
)
parser.add_argument(
    "--wandb-project-name",
    type=str,
    default='open-clip',
    help="Name of the project if logging with wandb.",
)
parser.add_argument(
    "--debug",
    default=False,
    action="store_true",
    help="If true, more information is logged."
)
parser.add_argument(
    "--copy-codebase",
    default=False,
    action="store_true",
    help="If true, we copy the entire base on the log directory, and execute from there."
)
parser.add_argument(
    "--horovod",
    default=False,
    action="store_true",
    help="Use horovod for distributed training."
)
parser.add_argument(
    "--ddp-static-graph",
    default=False,
    action='store_true',
    help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
)
parser.add_argument(
    "--no-set-device-rank",
    default=False,
    action="store_true",
    help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
)
parser.add_argument(
    "--seed", type=int, default=0, help="Default random seed."
)
parser.add_argument(
    "--grad-clip-norm", type=float, default=None, help="Gradient clip."
)
parser.add_argument(
    "--lock-text",
    default=False,
    action='store_true',
    help="Lock full text tower by disabling gradients.",
)
parser.add_argument(
    "--lock-text-unlocked-layers",
    type=int,
    default=0,
    help="Leave last n text tower layer groups unlocked.",
)
parser.add_argument(
    "--lock-text-freeze-layer-norm",
    default=False,
    action='store_true',
    help="Freeze BatchNorm running stats in text tower for any locked layers.",
)
parser.add_argument(
    "--log-every-n-steps",
    type=int,
    default=100,
    help="Log every n steps to tensorboard/console/wandb.",
)
parser.add_argument(
    "--coca-caption-loss-weight",
    type=float,
    default=2.0,
    help="Weight assigned to caption loss in CoCa."
)
parser.add_argument(
    "--coca-contrastive-loss-weight",
    type=float,
    default=1.0,
    help="Weight assigned to contrastive loss when training CoCa."
)
parser.add_argument(
    "--remote-sync",
    type=str,
    default=None,
    help="Optinoally sync with a remote path specified by this arg",
)
parser.add_argument(
    "--remote-sync-frequency",
    type=int,
    default=300,
    help="How frequently to sync to a remote directly if --remote-sync is not None.",
)
parser.add_argument(
    "--remote-sync-protocol",
    choices=["s3", "fsspec"],
    default="s3",
    help="How to do the remote sync backup if --remote-sync is not None.",
)
parser.add_argument(
    "--delete-previous-checkpoint",
    default=False,
    action="store_true",
    help="If true, delete previous checkpoint after storing a new one."
)
parser.add_argument(
    "--distill-model",
    default=None,
    help='Which model arch to distill from, if any.'
)
parser.add_argument(
    "--distill-pretrained",
    default=None,
    help='Which pre-trained weights to distill from, if any.'
)
parser.add_argument(
    "--siglip",
    default=None,
    help='Which pre-trained weights to distill from, if any.'
)
parser.add_argument(
    "--use-bnb-linear",
    default=None,
    help='Replace the network linear layers from the bitsandbytes library. '
    'Allows int8 training/inference, etc.'
)
parser.add_argument(
    "--da",
    default=True,
    action="store_true",
    help="If true, train/finetune degradation aware CLIP."
)
parser.add_argument(
    "--crop",
    default=True,
    action="store_true",
    help="If true, random crop the image for DaCLIP."
)














if __name__ == "__main__":

    # main(sys.argv[1:])

    arg = parser.parse_args()

    main(arg)