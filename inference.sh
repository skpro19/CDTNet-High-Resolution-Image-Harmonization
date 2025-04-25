#!/bin/bash

# Default values
MODEL_TYPE="CDTNet"
# Checkpoint logic moved below after parsing
INPUT_BASE_DIR="../output-backend" # Base directory for inputs/outputs relative to cdtNet
WARPING_DIR="${INPUT_BASE_DIR}/warping" # Input directory for warped images
MASK_DIR="${INPUT_BASE_DIR}/segmentation" # Input directory for masks
OUTPUT_DIR="${INPUT_BASE_DIR}/harmonize" # Output directory for harmonized images
TARGET_IMAGE="composited.png" # Default target image filename
DEFAULT_MASK_NAME="sam-mask.png" # Default mask name to look for
GPU="cuda:0"
CONFIG_PATH="./config.yml"
# Default high-quality settings (can be overridden by flags below)
LR=512
HR_H=2048
HR_W=2048
IS_SIM=true
ORIGINAL_SIZE=true
DEBUG=false
BINARIZE_MASK=false
CHECKPOINT_OVERRIDDEN=false # Track if user explicitly set a checkpoint

# --- Argument Parsing --- 
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --checkpoint) 
            CHECKPOINT="$2"
            CHECKPOINT_OVERRIDDEN=true # Mark checkpoint as overridden
            shift ;;
        --warping-dir) WARPING_DIR="$2"; shift ;;
        --mask-dir) MASK_DIR="$2"; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        --target-image) TARGET_IMAGE="$2"; shift ;; 
        --default-mask-name) DEFAULT_MASK_NAME="$2"; shift ;; 
        --gpu) GPU="$2"; shift ;;
        --config-path) CONFIG_PATH="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --hr_h) HR_H="$2"; shift ;;
        --hr_w) HR_W="$2"; shift ;;
        # Allow overriding default high-quality settings
        --no-is_sim) IS_SIM=false ;;
        --no-original-size) ORIGINAL_SIZE=false ;;
        --debug) DEBUG=true ;;
        --binarize-mask) BINARIZE_MASK=true ;;
        *) echo "Unknown option $1"; exit 1 ;;
    esac
    shift # past argument or value
done
# --- End Argument Parsing ---

# Set default checkpoint based on IS_SIM status *unless* overridden by user
if [ "$CHECKPOINT_OVERRIDDEN" = false ]; then
    echo "No checkpoint specified, determining default based on settings..."
    if [ "$IS_SIM" = true ] && [ -f "./CDTNet_sim_base256.pth" ]; then
        CHECKPOINT="./CDTNet_sim_base256.pth"
        echo "  - Using default sim checkpoint: $CHECKPOINT"
    elif [ "$IS_SIM" = true ]; then # IS_SIM is true but file not found
        echo "  - Warning: Defaulting to sim mode but CDTNet_sim_base256.pth not found."
        echo "           Please ensure the correct checkpoint is available or specify one with --checkpoint."
        echo "           Falling back to HAdobe5k_2048.pth for now."
        CHECKPOINT="./HAdobe5k_2048.pth"
    else # IS_SIM is false
        CHECKPOINT="./HAdobe5k_2048.pth"
        echo "  - Using default non-sim checkpoint: $CHECKPOINT"
    fi
else
    echo "User specified checkpoint: $CHECKPOINT"
    # Optional: Warn if user specified HAdobe5k but IS_SIM is true?
    # if [ "$IS_SIM" = true ] && [ "$CHECKPOINT" = "./HAdobe5k_2048.pth" ]; then
    #     echo "  - Warning: Running in sim mode (IS_SIM=true) but using HAdobe5k checkpoint."
    # fi
fi


# Check if required files and base directories exist
if [ ! -f "$CHECKPOINT" ]; then echo "Error: Checkpoint file not found: $CHECKPOINT"; exit 1; fi
if [ ! -d "$WARPING_DIR" ]; then echo "Error: Warping directory not found: $WARPING_DIR"; exit 1; fi
if [ ! -d "$MASK_DIR" ]; then echo "Error: Mask directory not found: $MASK_DIR"; exit 1; fi

# Create output base directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting CDTNet Harmonization (High-Quality Defaults Enabled)..."
echo "Target Image: $TARGET_IMAGE (using mask: $DEFAULT_MASK_NAME as fallback)"
echo "Warped Images Base Dir: $WARPING_DIR"
echo "Masks Base Dir: $MASK_DIR"
echo "Output Base Dir: $OUTPUT_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "GPU: $GPU"
echo "Resolution: ${HR_W}x${HR_H} (base: $LR)"
if [ "$IS_SIM" = true ]; then echo "Using CDTNet-sim mode"; fi
if [ "$ORIGINAL_SIZE" = true ]; then echo "Will resize outputs to original image sizes"; fi
if [ "$DEBUG" = true ]; then echo "Debug mode enabled"; fi
if [ "$BINARIZE_MASK" = true ]; then echo "Mask binarization enabled"; fi

# Find subdirectories
for SUBDIR in $(ls "$WARPING_DIR"); do
    INPUT_IMG_SUBDIR="$WARPING_DIR/$SUBDIR"
    INPUT_MASK_SUBDIR="$MASK_DIR/$SUBDIR"
    OUTPUT_SUBDIR="$OUTPUT_DIR/$SUBDIR"

    if [ -d "$INPUT_IMG_SUBDIR" ]; then
        echo "Processing subdirectory: $SUBDIR"

        if [ ! -d "$INPUT_MASK_SUBDIR" ]; then
            echo "Warning: Mask subdirectory not found for $SUBDIR: $INPUT_MASK_SUBDIR. Skipping."
            continue
        fi

        if [ ! -f "$INPUT_IMG_SUBDIR/$TARGET_IMAGE" ]; then
           echo "Warning: Target image '$TARGET_IMAGE' not found in $INPUT_IMG_SUBDIR. Skipping."
           continue
        fi

        if [ ! -f "$INPUT_MASK_SUBDIR/$TARGET_IMAGE" ] && [ ! -f "$INPUT_MASK_SUBDIR/$DEFAULT_MASK_NAME" ]; then
           echo "Warning: Neither target mask '$TARGET_IMAGE' nor default mask '$DEFAULT_MASK_NAME' found in $INPUT_MASK_SUBDIR. Skipping."
           continue
        fi

        mkdir -p "$OUTPUT_SUBDIR"

        # Build the command, passing target image and default mask name
        CMD="python scripts/inference.py \"$MODEL_TYPE\" \"$CHECKPOINT\" --images \"$INPUT_IMG_SUBDIR\" --masks \"$INPUT_MASK_SUBDIR\" --gpu \"$GPU\" --output-dir \"$OUTPUT_SUBDIR\" --target-image \"$TARGET_IMAGE\" --default-mask-name \"$DEFAULT_MASK_NAME\" --config-path \"$CONFIG_PATH\" --lr $LR --hr_h $HR_H --hr_w $HR_W"

        # Add optional flags (These now override the defaults)
        if [ "$IS_SIM" = true ]; then CMD="$CMD --is_sim"; else CMD="$CMD"; fi # Pass --is_sim only if true
        if [ "$ORIGINAL_SIZE" = true ]; then CMD="$CMD --original-size"; else CMD="$CMD"; fi # Pass --original-size only if true
        if [ "$DEBUG" = true ]; then CMD="$CMD --debug"; fi
        if [ "$BINARIZE_MASK" = true ]; then CMD="$CMD --binarize-mask"; fi

        echo "Running command for subdir $SUBDIR: $CMD"
        eval $CMD

        if [ $? -eq 0 ]; then
            echo "Harmonization successful for $SUBDIR. Copying input files..."
            cp -n "$INPUT_IMG_SUBDIR"/* "$OUTPUT_SUBDIR"/ > /dev/null 2>&1 || echo "Warning: Some files might not have been copied from $INPUT_IMG_SUBDIR"
        else
            echo "Error processing subdirectory $SUBDIR. Check logs. Skipping file copy."
        fi
    else
         echo "Skipping non-directory item: $SUBDIR in $WARPING_DIR"
    fi
done

echo "Harmonization finished. Outputs saved in $OUTPUT_DIR"

# Removed chmod +x command as it was causing issues when run from within cdtNet 