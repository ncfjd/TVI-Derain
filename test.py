import argparse
import os

import ast
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from rlp.dataset import DatasetTest
from rlp.models import model_utils
from rlp.utils import expand2square
from open_clip.factory import create_model_and_transforms

import open_clip
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


parser = argparse.ArgumentParser(description='Image deraining inference on GTAV-NightRain')

parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

parser.add_argument('--input_dir', default=r'D:\Song\Rain200H\train', type=str, help='Directory of test images')
parser.add_argument('--result_dir', default='D:/Python Code/RLP-main/RLP-main/logs/testresults', type=str, help='Directory for results')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')

parser.add_argument('--model_name', default='UNet_', type=str, help='arch')
parser.add_argument('--weights', default=r'D:\Python Code\RLP-main\RLP-main\logs\deraining\GTAV-NightRain\UNet_\models\model_epoch_253.pth', type=str, help='Path to weights')

# args only for Uformer
parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
parser.add_argument('--embed_dim', type=int, default=16, help='number of data loading workers')
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
parser.add_argument('--query_embed', action='store_true', default=False, help='query embedding for the decoder')
parser.add_argument('--dd_in', type=int, default=3, help='dd_in')

parser.add_argument('--tile', action='store_true', default=False, help='whether to tile for test image of large size')

##############
# args for clip
parser.add_argument('--model', type=str, default="ViT-B-32", help="Name of the vision backbone to use.")
parser.add_argument("--pretrained", default='', type=str,
                    help="Use a pretrained CLIP model weights with the specified tag or file path.", )
parser.add_argument("--precision",
                    choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
                    default="amp", help="Floating point precision.")
parser.add_argument("--torchscript", default=False, action='store_true',
                    help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'", )
parser.add_argument('--force-image-size', type=int, nargs='+', default=None, help='Override default image size')
parser.add_argument("--force-quick-gelu", default=False, action='store_true',
                    help="Force use of QuickGELU activation for non-OpenAI transformer models.", )
parser.add_argument("--force-patch-dropout", default=None, type=float,
                    help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper", )
parser.add_argument("--force-custom-text", default=False, action='store_true',
                    help="Force use of CustomTextCLIP model (separate text-tower).", )
parser.add_argument("--pretrained-image", default=False, action='store_true',
                    help="Load imagenet pretrained weights for image tower backbone if available.", )
parser.add_argument('--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override default image mean value of dataset')
parser.add_argument('--image-std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override default image std deviation of of dataset')
parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)

parser.add_argument('--daclip-model', type=str, default="daclip_ViT-B-32", help="Name of the vision backbone to use.")
parser.add_argument("--daclip-pretrained", default='daclip_ViT-B-32.pt', type=str,
                        help="Use a pretrained CLIP model weights with the specified tag or file path.", )
parser.add_argument('--csv_path', type=str, default='D:\Song\Rain200H/val.csv', help='dir of train data')
############
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


if __name__ == "__main__":


    args.arch = ''.join([x if x in args.model_name else '' for x in ['UNet', 'Uformer_T']])
    args.use_rlp  = 'RLP'  in args.model_name
    args.use_rpim = 'RPIM' in args.model_name

    ##########  clip Model ##########
    # device = init_distributed_device(args)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device='cpu',
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

    ######################################################

    model_restoration = model_utils.get_arch(args,model)

    model_utils.load_checkpoint(model_restoration,args.weights)    
    print("===>Testing using weights: ",args.weights)
    
    model_restoration.cuda()
    model_restoration.eval()
    
    test_dataset = DatasetTest(args.input_dir,args.csv_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False, pin_memory=False)
    
    result_dir = os.path.join(args.result_dir, args.model_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        for i, (input, filename,text) in enumerate(tqdm(test_loader)):
            input = input.cuda()

            if not args.tile:
                if 'Uformer' in args.arch:
                    b, _, h, w = input.size()
                    # Uformer accepts squared inputs
                    if not args.tile:
                        input, mask = expand2square(input)

                restored,_= model_restoration(input,text)

                if 'Uformer' in args.arch:
                    restored = torch.masked_select(restored, mask.bool()).reshape(b, 3, h, w)
                
            else:
                b, _, h, w = input.size()
                # for batch processing or large images, tiling it
                # currently used for large Uformer on GTAV-NightRain data
                tiles = []
                masks = []
                tile, mask = expand2square(input[:,:,:,:1280], factor=128)
                tiles.append(tile)
                masks.append(mask)
                tile, mask = expand2square(input[:,:,:,-1280:], factor=128)
                tiles.append(tile)
                masks.append(mask)

                restored = []
                for i in range(len(tiles)):
                    tile_restored, _ = model_restoration(tiles[i])
                    
                    tile_restored = torch.masked_select(tile_restored,(masks[i].bool())).reshape(b,3,h,1280)
                    restored.append(tile_restored)

                restored = torch.cat([restored[0][:,:,:,:960],restored[1][:,:,:,-960:]],3)
            
            restored = torch.clamp(restored, 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().numpy()
            for batch in range(len(restored)):
                restored_img = restored[batch]
                restored_img = np.uint8(restored_img * 255)
                restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(result_dir, filename[batch] + '.png'), restored_img)


