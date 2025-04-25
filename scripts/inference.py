import cv2
import torch
import numpy as np
import argparse
import os
import os.path as osp
from pathlib import Path
import sys
import yaml
from tqdm import tqdm
from easydict import EasyDict as edict

# Add project root to sys.path to allow imports
sys.path.insert(0, '.')
from iharm.model.base import CDTNet
from iharm.inference.predictor import Predictor
from iharm.inference.utils import load_model, find_checkpoint
from iharm.mconfigs import ALL_MCONFIGS
from iharm.utils.log import logger

def load_config_file(config_path, model_name=None, return_edict=False):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if 'SUBCONFIGS' in cfg:
        if model_name is not None and model_name in cfg['SUBCONFIGS']:
            cfg.update(cfg['SUBCONFIGS'][model_name])
        del cfg['SUBCONFIGS']

    return edict(cfg) if return_edict else cfg

def main():
    args, cfg = parse_args()

    # Enable debug mode for more verbose output
    if args.debug:
        import logging
        logger.setLevel(logging.DEBUG)

    # Load model
    device = torch.device(args.gpu)
    checkpoint_path = find_checkpoint(cfg.MODELS_PATH, args.checkpoint)
    model = load_model(args.model_type, checkpoint_path, verbose=True)

    # Set resolution
    model.set_resolution(args.hr_h, args.hr_w, args.lr, False)
    model.is_sim = args.is_sim
    logger.debug(f"Model settings: lr={args.lr}, hr_h={args.hr_h}, hr_w={args.hr_w}, is_sim={args.is_sim}")

    # Create predictor
    predictor = Predictor(model, device)

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Output directory: {output_dir}')

    # Construct path for the target image
    image_name = args.target_image
    image_path = osp.join(args.images, image_name)
    logger.info(f'Processing target image: {image_path}')

    # --- Process Single Target Image ---
    if not osp.exists(image_path):
        logger.error(f"Target image not found: {image_path}")
        sys.exit(1) # Exit if target image doesn't exist

    logger.debug(f"Loading image: {image_name}")
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        sys.exit(1)

    # Convert to RGB for model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logger.debug(f"Image shape: {image.shape}, dtype: {image.dtype}")

    # Store original size if needed
    image_size = image.shape

    # Resize image to target resolution
    image = cv2.resize(image, (args.hr_w, args.hr_h), cv2.INTER_LINEAR)
    logger.debug(f"Resized image shape: {image.shape}")

    # Load and process mask - Try target name first, then default mask name
    mask_path_target = osp.join(args.masks, image_name) # Mask with same name as target
    mask_path_default = osp.join(args.masks, args.default_mask_name) # Default mask name (e.g., sam-mask.png)
    mask_path = None

    if osp.exists(mask_path_target):
        mask_path = mask_path_target
        logger.debug(f"Found mask matching target image name: {mask_path}")
    elif osp.exists(mask_path_default):
        mask_path = mask_path_default
        logger.debug(f"Using default mask name: {mask_path}")
    else:
        logger.error(f"Could not find mask for {image_name}. Looked for {mask_path_target} and {mask_path_default}")
        sys.exit(1) # Exit if no suitable mask found

    logger.debug(f"Loading mask from: {mask_path}")
    mask_image = cv2.imread(mask_path)
    if mask_image is None:
        logger.error(f"Could not load mask: {mask_path}")
        sys.exit(1)

    mask_image = mask_image.astype(np.float32) / 255
    mask_image = cv2.resize(mask_image, (args.hr_w, args.hr_h), cv2.INTER_LINEAR)
    mask = mask_image[:, :, 0]  # Use first channel as mask
    logger.debug(f"Mask shape: {mask.shape}, min: {mask.min()}, max: {mask.max()}")

    # Binarize mask if needed
    if args.binarize_mask:
        mask[mask <= 0.39] = 0  # ~100/255
        mask[mask > 0.39] = 1
        logger.debug(f"Binarized mask, min: {mask.min()}, max: {mask.max()}")

    # Run prediction
    try:
        with torch.no_grad():
            logger.debug("Running direct prediction mode (return_numpy=True)")
            prediction = predictor.predict(image, mask)
            logger.debug(f"Prediction type: {type(prediction)}, shape: {prediction.shape if hasattr(prediction, 'shape') else 'unknown'}")
    except Exception as e:
        logger.error(f"Error during prediction for {image_name}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Process prediction output
    try:
        final_output_np = None
        if isinstance(prediction, np.ndarray):
            logger.debug(f"Processing numpy prediction, shape: {prediction.shape}")
            final_output_np = prediction
        elif torch.is_tensor(prediction):
            logger.debug(f"Processing tensor prediction, shape: {prediction.shape}")
            if len(prediction.shape) == 4: final_output_np = prediction[0].permute(1, 2, 0).cpu().numpy()
            elif len(prediction.shape) == 3:
                if prediction.shape[0] == 3: final_output_np = prediction.permute(1, 2, 0).cpu().numpy()
                else: final_output_np = prediction.cpu().numpy()
            else: final_output_np = prediction.cpu().numpy()
            logger.debug(f"Converted tensor to numpy, shape: {final_output_np.shape if final_output_np is not None else 'None'}")
        else:
            logger.error(f"Unknown prediction type: {type(prediction)}")
            sys.exit(1)

        if final_output_np is None:
            logger.error("Could not extract final output from prediction.")
            sys.exit(1)

        # Resize back to original if requested
        if args.original_size and image_size[:2] != final_output_np.shape[:2]:
            logger.debug(f"Resizing output back to {image_size[1]}x{image_size[0]}")
            final_output_np = cv2.resize(final_output_np, (image_size[1], image_size[0]), interpolation=cv2.INTER_LANCZOS4)

    except Exception as e:
        logger.error(f"Error processing prediction for {image_name}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save the final harmonized image as harmonized.png
    try:
        logger.debug(f"Saving final output, shape: {final_output_np.shape}, min: {final_output_np.min()}, max: {final_output_np.max()}")

        if final_output_np.max() <= 1.0 and final_output_np.min() >= 0.0:
            final_output_np = final_output_np * 255

        if final_output_np.shape[-1] == 3:
            final_output_bgr = cv2.cvtColor(final_output_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            final_output_bgr = final_output_np.astype(np.uint8)

        # Save as harmonized.png
        output_final_path = output_dir / "harmonized.png"
        logger.debug(f"Saving final output to {output_final_path}")
        cv2.imwrite(str(output_final_path), final_output_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3]) # Use PNG for potentially better quality

        logger.info(f"Saved harmonized output for {image_name} to {output_final_path}")

    except Exception as e:
        logger.error(f"Error saving output for {image_name}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    logger.info(f"Harmonization processing finished successfully for {args.images}. Output saved in {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run CDTNet harmonization inference for a specific target image.")
    parser.add_argument('model_type', choices=ALL_MCONFIGS.keys(), help='Type of the model')
    parser.add_argument('checkpoint', type=str,
                        help='The path to the model checkpoint. '
                             'This can be a relative path (relative to cfg.MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
    parser.add_argument('--images', type=str, required=True,
                        help='Path to directory containing the input target image.')
    parser.add_argument('--masks', type=str, required=True,
                        help='Path to directory containing the corresponding mask.')
    parser.add_argument('--gpu', type=str, default='cuda:0',
                        help='GPU device to use (e.g., cuda:0 or cpu)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save the final harmonized output image (harmonized.png).')
    parser.add_argument('--target-image', type=str, default='composited.png',
                        help='Filename of the target image within the --images directory.')
    parser.add_argument('--default-mask-name', type=str, default='sam-mask.png',
                        help='Default filename to look for in the --masks directory if a mask matching --target-image is not found.')
    parser.add_argument('--lr', type=int, default=512,
                        help='Base resolution for UNet (low-res)')
    parser.add_argument('--hr_h', type=int, default=2048,
                        help='Target height resolution')
    parser.add_argument('--hr_w', type=int, default=2048,
                        help='Target width resolution')
    parser.add_argument('--is_sim', action='store_true', default=False,
                        help='Whether to use CDTNet-sim (LUT only). Default: False')
    parser.add_argument('--original-size', action='store_true', default=False,
                        help='Resize predicted image back to the original size.')
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='The path to the config file.')
    parser.add_argument('--binarize-mask', action='store_true', default=False,
                        help='Binarize mask before processing. Default: False')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable detailed debug output')

    args = parser.parse_args()
    cfg = load_config_file(args.config_path, return_edict=True)

    if not hasattr(cfg, 'MODELS_PATH'):
        cfg.MODELS_PATH = './'

    return args, cfg

if __name__ == "__main__":
    main() 