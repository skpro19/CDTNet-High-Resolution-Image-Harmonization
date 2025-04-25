#!/bin/bash

# Default values
MODEL_TYPE="CDTNet"
CHECKPOINT="./HAdobe5k_2048.pth"  # Example checkpoint
IMAGE_DIR="./predict_images" # Default image directory
MASK_DIR="./predict_masks"   # Default mask directory
GPU="cuda:0"
OUTPUT_DIR="./module_outputs"
CONFIG_PATH="./config.yml"
LR=512
HR_H=2048
HR_W=2048
IS_SIM=false
ORIGINAL_SIZE=false
ADDITIONAL=false
DEBUG=false
BINARIZE_MASK=false

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift # past argument
            shift # past value
            ;;
        --image-dir)
            IMAGE_DIR="$2"
            shift # past argument
            shift # past value
            ;;
        --mask-dir)
            MASK_DIR="$2"
            shift # past argument
            shift # past value
            ;;
        --gpu)
            GPU="$2"
            shift # past argument
            shift # past value
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift # past argument
            shift # past value
            ;;
        --config-path)
            CONFIG_PATH="$2"
            shift # past argument
            shift # past value
            ;;
        --lr)
            LR="$2"
            shift # past argument
            shift # past value
            ;;
        --hr_h)
            HR_H="$2"
            shift # past argument
            shift # past value
            ;;
        --hr_w)
            HR_W="$2"
            shift # past argument
            shift # past value
            ;;
        --is_sim)
            IS_SIM=true
            shift # past argument
            ;;
        --original-size)
            ORIGINAL_SIZE=true
            shift # past argument
            ;;
        --additional)
            ADDITIONAL=true
            shift # past argument
            ;;
        --debug)
            DEBUG=true
            shift # past argument
            ;;
        --binarize-mask)
            BINARIZE_MASK=true
            shift # past argument
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Apply additional flag settings (only if individual flags weren't explicitly set)
if [ "$ADDITIONAL" = true ]; then
    echo "Enabling all additional options with default values"
    
    # If checkpoint is HAdobe5k_2048.pth or was not explicitly set, use CDTNet_sim_base256.pth for --additional
    if [ "$CHECKPOINT" = "./HAdobe5k_2048.pth" ]; then
        if [ -f "./CDTNet_sim_base256.pth" ]; then
            CHECKPOINT="./CDTNet_sim_base256.pth"
            echo "  - Switched to sim-based checkpoint: $CHECKPOINT"
        fi
    fi
    
    # Set high-quality defaults
    LR=512
    HR_H=2048
    HR_W=2048
    IS_SIM=true
    ORIGINAL_SIZE=true
    DEBUG=true
    BINARIZE_MASK=true
    
    echo "  - Using resolution: ${HR_W}x${HR_H} (base: $LR)"
    echo "  - Enabled CDTNet-sim mode"
    echo "  - Will resize outputs to original image sizes"
    echo "  - Enabled debug output for verbose logging"
    echo "  - Enabled mask binarization"
fi

# Check if required files exist
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory not found: $IMAGE_DIR"
    exit 1
fi
if [ ! -d "$MASK_DIR" ]; then
    echo "Error: Mask directory not found: $MASK_DIR"
    exit 1
fi
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Warning: Config file not found: $CONFIG_PATH"
    echo "Will use default configuration."
fi

echo "Running module visualization for directory..."
echo "Checkpoint: $CHECKPOINT"
echo "Image Dir: $IMAGE_DIR"
echo "Mask Dir: $MASK_DIR"
echo "GPU: $GPU"
echo "Output Dir: $OUTPUT_DIR"
echo "Config Path: $CONFIG_PATH"
echo "Resolution: ${HR_W}x${HR_H} (base: $LR)"
if [ "$IS_SIM" = true ]; then
    echo "Using CDTNet-sim mode"
fi
if [ "$ORIGINAL_SIZE" = true ]; then
    echo "Will resize outputs to original image sizes"
fi
if [ "$DEBUG" = true ]; then
    echo "Debug mode enabled"
fi
if [ "$BINARIZE_MASK" = true ]; then
    echo "Mask binarization enabled"
fi

# Build command with required arguments
CMD="python scripts/test_cdtNet_modules.py \"$MODEL_TYPE\" \"$CHECKPOINT\" --images \"$IMAGE_DIR\" --masks \"$MASK_DIR\" --gpu \"$GPU\" --output-dir \"$OUTPUT_DIR\" --config-path \"$CONFIG_PATH\" --lr $LR --hr_h $HR_H --hr_w $HR_W"

# Add optional flags
if [ "$IS_SIM" = true ]; then
    CMD="$CMD --is_sim"
fi
if [ "$ORIGINAL_SIZE" = true ]; then
    CMD="$CMD --original-size"
fi
if [ "$DEBUG" = true ]; then
    CMD="$CMD --debug"
fi
if [ "$BINARIZE_MASK" = true ]; then
    CMD="$CMD --binarize-mask"
fi

# Execute the command
echo "Running command: $CMD"
eval $CMD

echo "Script finished. Outputs saved in $OUTPUT_DIR" 