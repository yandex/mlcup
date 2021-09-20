# generic imports
import click
from omegaconf import OmegaConf

# torch imports
import torch

# custom imports
from i2t.system import I2T



@click.command()
@click.option('--train_ckpt_file', help='Path to original checkpoint')
@click.option('--inference_ckpt_file', help='Path which resulting checkpoint should be saved to')
@click.option('--embeddings_to_fp16', is_flag=True, help='Cast embeds to fp16')
@click.option('--model_to_fp16', is_flag=True, help='Cast whole model to fp16')
def main(
    train_ckpt_file: str,
    inference_ckpt_file: str,
    model_to_fp16: bool,
    embeddings_to_fp16: bool
):
    ckpt = torch.load(train_ckpt_file, map_location='cpu')
    del ckpt['optimizer_states']
    del ckpt['lr_schedulers']
    del ckpt['callbacks']
    ckpt['hyper_parameters']['model']['text']['args']['encoder']['args']['pretrained_bpemb_embeddings'] = False
    cfg = OmegaConf.create(ckpt['hyper_parameters'])
    model = I2T(config=cfg)
    model.load_state_dict(ckpt['state_dict'])
    if model_to_fp16:
        model = model.half()
    elif embeddings_to_fp16:
        model.encoders['text'].encoder.embedding = \
            model.encoders['text'].encoder.embedding.half()
    new_state_dict = model.state_dict()
    ckpt['state_dict'] = new_state_dict
    torch.save(ckpt, inference_ckpt_file)


if __name__ == '__main__':
    main()