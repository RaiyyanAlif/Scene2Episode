#!/bin/bash
set -e

apt-get update
apt-get install -y ffmpeg

ffmpeg -version
ffprobe -version
