#!/bin/bash
set -e

I2A_S1=https://drive.google.com/drive/folders/1uNQucxKvYcuv8UwWKcDSsUk9K_IzskxY?usp=drive_link
I2A_S2=https://drive.google.com/drive/u/1/folders/1w0ExZ67RcF5XVlJ87VTrSLHYa0JUo5wF

if [ ! -d "$HOME/Documents/I2A_S1" ]; then
    echo "$HOME/Documents/I2A_S1 found. Download the drive folder at:"
    echo "      $I2A_S1"
    echo ""
else
    ln -sfn $HOME/Documents/I2A_S1/TAI/ ./UNI_COURSES//
    ln -sfn $HOME/Documents/I2A_S1/ML/ ./UNI_COURSES/
fi

if [ ! -d "$HOME/Documents/I2A_S2" ]; then
    echo "$HOME/Documents/I2A_S2 found. Download the drive folder at:"
    echo "      $I2A_S2"
    echo ""
else
    ln -sfn $HOME/Documents/I2A_S2/DL/ ./UNI_COURSES/
fi
