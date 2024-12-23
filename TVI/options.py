import argparse
import ast
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
def parse_options():
    """docstring for training configuration"""
    parser = argparse.ArgumentParser(description='Image deraining training on GTAV-NightRain')

    # args for arch selection
    parser.add_argument('--mode', type=str, default ='deraining',  help='image restoration mode')
    parser.add_argument('--arch', type=str, default ='UNet',  help='archtechture')
    parser.add_argument('--use_rlp', action='store_true', default=False, help='whether to use RLP')
    parser.add_argument('--use_rpim', action='store_true', default=False, help='whether to use RPIM')

    # args for training
    parser.add_argument('--train_dir', type=str, default ='D:\Song\Rain13k/train',  help='dir of train data')
    parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--train_workers', type=int, default=0, help='train_dataloader workers')
    parser.add_argument('--gpu', type=str, default='0', help='GPUs')

    parser.add_argument('--nepoch', type=int, default=300, help='training epochs')
    parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
    parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay of')        
    parser.add_argument('--warmup', action='store_true', default=True, help='warmup')
    parser.add_argument('--warmup_epochs', type=int,default=3, help='epochs for warmup')

    # args for logging
    parser.add_argument('--save_dir', type=str, default ='./logs/',  help='save dir')
    parser.add_argument('--save_images', action='store_true',default=True)
    parser.add_argument('--env', type=str, default ='_',  help='env')
    parser.add_argument('--dataset', type=str, default ='GTAV-NightRain')
    parser.add_argument('--checkpoint', type=int, default=1, help='epochs to save checkpoint')

    # args for resuming training
    parser.add_argument('--resume', action='store_true',default=True)
    #parser.add_argument('--pretrain_weights',type=str, default='./log/Uformer_B/models/model_best.pth', help='path of pretrained_weights')
    parser.add_argument('--pretrain_weights', type=str, default=r'D:\Python Code\RLP-main\RLP-main\logs\deraining\GTAV-NightRain\UNet_\models\model_epoch_55.pth',
                        help='path of pretrained_weights')

    # args for Uformer
    parser.add_argument('--dd_in', type=int, default=3, help='dd_in')
    parser.add_argument('--norm_layer', type=str, default ='nn.LayerNorm', help='normalize layer in transformer')
    parser.add_argument('--embed_dim', type=int, default=16, help='dim of emdeding features')
    parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')
    parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
    parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
    parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')
    parser.add_argument('--modulator', action='store_true', default=False, help='multi-scale modulator')

    #############################################
    # args for clip
    parser.add_argument('--model',type=str,default="ViT-B-32",help="Name of the vision backbone to use.")
    parser.add_argument("--pretrained",default='laion2b_s34b_b79k',type=str,help="Use a pretrained CLIP model weights with the specified tag or file path.",)
    parser.add_argument("--precision",choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],default="amp",help="Floating point precision.")
    parser.add_argument("--torchscript",default=False,action='store_true',help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",)
    parser.add_argument('--force-image-size', type=int, nargs='+', default=None,help='Override default image size')
    parser.add_argument("--force-quick-gelu",default=False,action='store_true',help="Force use of QuickGELU activation for non-OpenAI transformer models.",)
    parser.add_argument("--force-patch-dropout",default=None,type=float,help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",)
    parser.add_argument("--force-custom-text",default=False,action='store_true',help="Force use of CustomTextCLIP model (separate text-tower).",)
    parser.add_argument("--pretrained-image",default=False,action='store_true',help="Load imagenet pretrained weights for image tower backbone if available.",)
    parser.add_argument('--image-mean', type=float, nargs='+', default=None, metavar='MEAN',help='Override default image mean value of dataset')
    parser.add_argument('--image-std', type=float, nargs='+', default=None, metavar='STD',help='Override default image std deviation of of dataset')
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
    parser.add_argument('--csv_path', type=str, default=r'D:\Song\Rain13k\train.csv', help='dir of train data')

    # parser.add_argument('--daclip-model', type=str, default="daclip_ViT-B-32", help="Name of the vision backbone to use.")
    # parser.add_argument("--daclip-pretrained", default='daclip_ViT-B-32.pt', type=str,
    #                     help="Use a pretrained CLIP model weights with the specified tag or file path.", )


    ##############################################
    
    return parser
