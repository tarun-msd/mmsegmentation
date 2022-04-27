from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

import pandas as pd
import boto3
import argparse
import os
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from multiprocessing import Pool
import torch
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
import urllib.request 
from joblib import Parallel, delayed
import numpy as np
import cv2


s3_client = boto3.client("s3")

def upload_image_to_s3(pil_image, s3_key):

    in_mem_file = BytesIO()
    pil_image.save(in_mem_file, format=pil_image.format)
    in_mem_file.seek(0)

    s3_client.upload_fileobj(
        in_mem_file,
        "msd-cvteam-apse",
        s3_key,
        ExtraArgs={
            "ACL": "public-read",
            "ContentType": "image/png"
        }
    )

def process_one_image(rec, folder):

    try:        
        mask_rgb = Image.open(rec["mask_rgb"])
        mask_rgb_identifier = rec['mask_rgb'].split("/")[-1]
        s3_mask_rgb_key = "Meli/img_correction/fg_segment/{}/mask_rgb/{}".format(folder, mask_rgb_identifier)
        upload_image_to_s3(mask_rgb, s3_mask_rgb_key)
        
        s3_mask_rgb_url = "https://msd-cvteam-apse.s3-ap-southeast-1.amazonaws.com/{}".format(s3_mask_rgb_key)
        return {
            "id" : rec['image_name'],
            "image_url" : rec['image_url'],
            "mask_rgb" : s3_mask_rgb_url,
            "mask_rgb_identifier" : mask_rgb_identifier.split(".")[0],
            "upload": "success"
        }
    except:
        print("ERROR")
        return {
            "upload": "error"
        }

def apply_mask(rec, mask, save_dir):
        req = urllib.request.urlopen(rec['image_url'])
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        temp_img = cv2.imdecode(arr, -1)
        mask_rgb = np.stack((temp_img[:,:,0],temp_img[:,:,1],temp_img[:,:,2],mask * 255),axis=2)
        assert mask_rgb.shape[-1] == 4
        image_id = rec['image_name'].split('.')[0] if '.' in rec['image_name'] else rec['image_name']
        mask_save_path = os.path.join(save_dir, 'mask_rgb')
        save_mask_rgb = '{}/{}_mask_rgb.png'.format(mask_save_path, image_id)
        cv2.imwrite(save_mask_rgb, mask_rgb)
        return save_mask_rgb

def inference(rec, model, save_dir):
    img_id = rec['image_name'] if rec['image_name'].lower().endswith(('.png', '.jpg', '.jpeg')) else rec['image_name'] + '.jpg'
    
    url = rec['image_url']

    image_save_path = os.path.join(save_dir, 'images')
    os.makedirs(image_save_path, exist_ok=True)

    save_path = os.path.join(image_save_path,img_id)
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)

    result = inference_segmentor(model, save_path)
    mask = result[0]

    save_mask_rgb = apply_mask(rec, mask, save_dir)
    assert len(mask.shape) == 2
#     rec.update({'mask_rgb' : save_mask_rgb})
    
#     upload = process_one_image(rec)
    
    return save_mask_rgb


def main(input_csv, config, checkpoint_path, identifier, save_dir, device = 'cuda:0'):
    
    model = init_segmentor(config, checkpoint_path, device=device)

    df = pd.read_csv(input_csv)
    df_lis = df.to_dict('records')
    
    mask_paths = Parallel(n_jobs=12, backend="threading")(
    delayed(inference)(rec, model, save_dir)
    for rec in tqdm(df_lis)
)
    df['mask_rgb'] = mask_paths
    df_lis = df.to_dict('records')
    
    result = Parallel(n_jobs=12, backend="threading")(
    delayed(process_one_image)(f, identifier)
    for f in tqdm(df_lis)
)
    rdf = pd.DataFrame(result)
    rdf.to_csv(os.path.join(save_dir, '{}.csv'.format(identifier)), index = False)

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True, default='/media/train_hdd5/tarun/data/testset/meli_fashion_testset.csv',help='path for data')
    parser.add_argument('--ckpt', type=str,required=True, default=None, help= 'checkpoint file')
    parser.add_argument('--config', type=str,required=True, default=None, help= 'config py file')
    parser.add_argument('--identifier', type=str, required=True, default=None, help= 'inf run identifier')
    parser.add_argument('--device', type=str, default='cuda:0',help='inference device')
    parser.add_argument('--save_dir', type=str, required=True, help='path for saving inference results')
    args = parser.parse_args()
    main(args.input_csv, args.config, args.ckpt, args.identifier, args.save_dir, args.device)