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

    # Prepare output directory and subdirectories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_dir / "final"
    lut_dir = output_dir / "lut"
    unet_dir = output_dir / "unet"
    final_dir.mkdir(exist_ok=True)
    lut_dir.mkdir(exist_ok=True)
    unet_dir.mkdir(exist_ok=True)

    # Get list of images
    image_names = os.listdir(args.images)
    logger.info(f'Processing {len(image_names)} images from {args.images}')
    logger.info(f'Saving outputs to {output_dir}')

    for image_name in tqdm(image_names):
        logger.debug(f"Processing image: {image_name}")
        # Load image
        image_path = osp.join(args.images, image_name)
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            continue
            
        # Convert to RGB for model
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logger.debug(f"Image shape: {image.shape}, dtype: {image.dtype}")
        
        # Store original size if needed
        image_size = image.shape
        
        # Resize image to target resolution
        image = cv2.resize(image, (args.hr_w, args.hr_h), cv2.INTER_LINEAR)
        logger.debug(f"Resized image shape: {image.shape}")

        # Load and process mask
        # Try both naming conventions: same as image or image_stem.png
        mask_path = osp.join(args.masks, image_name)
        if not os.path.exists(mask_path):
            mask_path = osp.join(args.masks, ''.join(image_name.split('.')[:-1]) + '.png')
        
        logger.debug(f"Looking for mask at: {mask_path}")
        mask_image = cv2.imread(mask_path)
        if mask_image is None:
            logger.warning(f"Could not load mask: {mask_path}")
            continue
            
        mask_image = mask_image.astype(np.float32) / 255
        mask_image = cv2.resize(mask_image, (args.hr_w, args.hr_h), cv2.INTER_LINEAR)
        mask = mask_image[:, :, 0]  # Use first channel as mask
        logger.debug(f"Mask shape: {mask.shape}, min: {mask.min()}, max: {mask.max()}")
        
        # Binarize mask if needed (as in predict_for_dir.py)
        if args.binarize_mask:
            mask[mask <= 0.39] = 0  # ~100/255
            mask[mask > 0.39] = 1
            logger.debug(f"Binarized mask, min: {mask.min()}, max: {mask.max()}")

        # Run prediction
        try:
            with torch.no_grad():
                # Set expected return type
                if args.extract_intermediates:
                    logger.debug("Requesting intermediate results (return_numpy=False)")
                    prediction = predictor.predict(image, mask, return_numpy=False)
                    logger.debug(f"Prediction type: {type(prediction)}")
                    if isinstance(prediction, dict):
                        logger.debug(f"Prediction keys: {prediction.keys()}")
                    elif torch.is_tensor(prediction):
                        logger.debug(f"Prediction shape: {prediction.shape}")
                else:
                    logger.debug("Using direct prediction mode (return_numpy=True)")
                    prediction = predictor.predict(image, mask)
                    logger.debug(f"Prediction type: {type(prediction)}, shape: {prediction.shape if hasattr(prediction, 'shape') else 'unknown'}")
        except Exception as e:
            logger.error(f"Error during prediction for {image_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Handle different output formats
        try:
            final_output_np = None
            lut_output_np = None
            unet_output_np = None
            
            if isinstance(prediction, dict) and 'images' in prediction:
                # Handle dictionary format with intermediate results
                logger.debug("Processing dictionary prediction format")
                if 'images' in prediction:
                    final_tensor = prediction['images']
                    if isinstance(final_tensor, list):
                        final_tensor = final_tensor[0]
                    if torch.is_tensor(final_tensor):
                        final_output_np = final_tensor.permute(1, 2, 0).cpu().numpy()
                        logger.debug(f"Got final output tensor, shape: {final_output_np.shape}")
                
                if 'lut_images' in prediction:
                    lut_tensor = prediction['lut_images']
                    if isinstance(lut_tensor, list) and len(lut_tensor) > 0:
                        lut_tensor = lut_tensor[0]
                    if torch.is_tensor(lut_tensor):
                        lut_output_np = lut_tensor.permute(1, 2, 0).cpu().numpy()
                        logger.debug(f"Got LUT output tensor, shape: {lut_output_np.shape}")
                
                if 'base_images' in prediction:
                    unet_tensor = prediction['base_images']
                    if isinstance(unet_tensor, list) and len(unet_tensor) > 0:
                        unet_tensor = unet_tensor[0]
                    if torch.is_tensor(unet_tensor):
                        unet_output_np = unet_tensor.permute(1, 2, 0).cpu().numpy()
                        logger.debug(f"Got UNet output tensor, shape: {unet_output_np.shape}")
            
            elif torch.is_tensor(prediction):
                # Handle tensor output directly
                logger.debug(f"Processing tensor prediction, shape: {prediction.shape}")
                if len(prediction.shape) == 4:  # NCHW format
                    final_output_np = prediction[0].permute(1, 2, 0).cpu().numpy()
                elif len(prediction.shape) == 3:  # Could be CHW or HWC format
                    if prediction.shape[0] == 3 or prediction.shape[2] == 3:
                        # Check which dimension is the channel dimension (likely 0 or 2)
                        if prediction.shape[0] == 3:  # CHW format
                            final_output_np = prediction.permute(1, 2, 0).cpu().numpy()
                        else:  # HWC format
                            final_output_np = prediction.cpu().numpy()
                    else:
                        logger.warning(f"Unexpected tensor shape: {prediction.shape}")
                        final_output_np = prediction.cpu().numpy()
                logger.debug(f"Converted tensor to numpy, shape: {final_output_np.shape if final_output_np is not None else 'None'}")
            
            elif isinstance(prediction, np.ndarray):
                # Handle numpy array output directly
                logger.debug(f"Processing numpy prediction, shape: {prediction.shape}")
                final_output_np = prediction
            
            else:
                logger.warning(f"Unknown prediction type: {type(prediction)}")
                continue
            
            # If we still don't have a final output, skip this image
            if final_output_np is None:
                logger.warning("Could not extract final output from prediction.")
                continue
                
            # Resize back to original if requested
            if args.original_size and image_size != final_output_np.shape:
                logger.debug(f"Resizing outputs back to {image_size[1]}x{image_size[0]}")
                final_output_np = cv2.resize(final_output_np, (image_size[1], image_size[0]))
                if lut_output_np is not None:
                    lut_output_np = cv2.resize(lut_output_np, (image_size[1], image_size[0]))
                if unet_output_np is not None:
                    unet_output_np = cv2.resize(unet_output_np, (image_size[1], image_size[0]))
            
        except Exception as e:
            logger.error(f"Error processing prediction for {image_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Save images
        try:
            # Final output saved as jpg with high quality
            logger.debug(f"Saving final output, shape: {final_output_np.shape}, min: {final_output_np.min()}, max: {final_output_np.max()}")
            
            # Scale values to 0-255 range if needed
            if final_output_np.max() <= 1.0 and final_output_np.min() >= 0.0:
                final_output_np = final_output_np * 255
                
            # RGB to BGR conversion for OpenCV
            if final_output_np.shape[-1] == 3:  # If it's an RGB image
                final_output_bgr = cv2.cvtColor(final_output_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
            else:
                final_output_bgr = final_output_np.astype(np.uint8)
                
            output_final_path = final_dir / image_name
            logger.debug(f"Saving final output to {output_final_path}")
            cv2.imwrite(str(output_final_path), final_output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
            logger.info(f"Saved final output for {image_name}")

            # Save intermediate outputs if available
            if lut_output_np is not None:
                logger.debug(f"Saving LUT output, shape: {lut_output_np.shape}")
                if lut_output_np.max() <= 1.0:
                    lut_output_np = lut_output_np * 255
                lut_output_bgr = cv2.cvtColor(lut_output_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
                output_lut_path = lut_dir / f"{os.path.splitext(image_name)[0]}.png"
                cv2.imwrite(str(output_lut_path), lut_output_bgr)
                logger.info(f"Saved LUT output for {image_name}")
                
            if unet_output_np is not None:
                logger.debug(f"Saving UNet output, shape: {unet_output_np.shape}")
                if unet_output_np.max() <= 1.0:
                    unet_output_np = unet_output_np * 255
                unet_output_bgr = cv2.cvtColor(unet_output_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
                output_unet_path = unet_dir / f"{os.path.splitext(image_name)[0]}.png"
                cv2.imwrite(str(output_unet_path), unet_output_bgr)
                logger.info(f"Saved UNet output for {image_name}")
            
        except Exception as e:
            logger.error(f"Error saving outputs for {image_name}: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"Processing finished. Outputs saved in subdirectories under {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize CDTNet intermediate outputs for a directory.")
    parser.add_argument('model_type', choices=ALL_MCONFIGS.keys(), help='Type of the model')
    parser.add_argument('checkpoint', type=str, 
                        help='The path to the model checkpoint. '
                             'This can be a relative path (relative to cfg.MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
    parser.add_argument('--images', type=str, required=True,
                        help='Path to directory with .jpg images to get predictions for.')
    parser.add_argument('--masks', type=str, required=True,
                        help='Path to directory with .png binary masks for images.')
    parser.add_argument('--gpu', type=str, default='cuda:0', 
                        help='GPU device to use (e.g., cuda:0 or cpu)')
    parser.add_argument('--output-dir', type=str, default='./module_outputs', 
                        help='Directory to save the output images')
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
    parser.add_argument('--extract-intermediates', action='store_true', default=True,
                        help='Extract intermediate results (LUT and UNet). Default: True')
    parser.add_argument('--binarize-mask', action='store_true', default=False,
                        help='Binarize mask like in predict_for_dir.py. Default: False')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable detailed debug output')

    args = parser.parse_args()
    cfg = load_config_file(args.config_path, return_edict=True)
    
    # Ensure MODELS_PATH exists in config
    if not hasattr(cfg, 'MODELS_PATH'):
        cfg.MODELS_PATH = './'
    
    return args, cfg


if __name__ == "__main__":
    main() 