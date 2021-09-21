#!/bin/sh

python train_ft.py \
	--snapshot-dir ./snapshots/GTA_TO_CITY \
	--restore-from ./snapshots/GTA_TO_CITY/GTA5_200000.pth \
	--drop 0.2 \
	--warm-up 5000 \
	--batch-size 1 \
	--learning-rate 1e-4 \
	--crop-size 512,256 \
	--lambda-seg 0.5 \
	--lambda-adv-target1 0 \
	--lambda-adv-target2 0 \
	--lambda-me-target 0 \
	--lambda-kl-target 0 \
	--norm-style gn \
	--class-balance \
	--only-hard-label 80 \
	--max-value 7 \
	--gpu-ids 0 \
	--often-balance \
	--use-se \
	--input-size 1280,640 \
	--train_bn \
	--autoaug False
