#!/bin/sh
./train.py --epochs 100 --optimizer Adam --lr 0.001 --deterministic --compress schedule-faceid.yaml --model ai85faceidnet --dataset FaceIDRot90 --batch-size 100 --device MAX78000 --regression --print-freq 250 "$@"
./train.py --epochs 100 --optimizer Adam --lr 0.001 --deterministic --compress schedule-faceid.yaml --model ai85faceidnetwide --dataset FaceIDRot90 --batch-size 100 --device MAX78000 --regression --print-freq 250 "$@"
./train.py --epochs 100 --optimizer Adam --lr 0.001 --deterministic --compress schedule-faceid.yaml --model ai85faceidnetv2 --dataset FaceIDRot90 --batch-size 100 --device MAX78000 --regression --print-freq 250 "$@"
./train.py --epochs 100 --optimizer Adam --lr 0.001 --deterministic --compress schedule-faceid.yaml --model ai85faceidnetv2bn --dataset FaceIDRot90 --batch-size 100 --device MAX78000 --regression --print-freq 250 --use-bias "$@"
./train.py --epochs 100 --optimizer Adam --lr 0.001 --deterministic --compress schedule-faceid.yaml --model ai85faceidnetwidebn --dataset FaceIDRot90 --batch-size 100 --device MAX78000 --regression --print-freq 250 --use-bias "$@"
./train.py --epochs 100 --optimizer Adam --lr 0.001 --deterministic --compress schedule-faceid.yaml --model ai85faceidnetv3 --dataset FaceIDRot90 --batch-size 100 --device MAX78000 --regression --print-freq 250 "$@"
./train.py --epochs 100 --optimizer Adam --lr 0.001 --deterministic --compress schedule-faceid.yaml --model ai85faceidnetfire --dataset FaceIDRot90 --batch-size 100 --device MAX78000 --regression --print-freq 250 "$@"
./train.py --epochs 100 --optimizer Adam --lr 0.001 --deterministic --compress schedule-faceid.yaml --model ai85faceidnetres --dataset FaceIDRot90 --batch-size 100 --device MAX78000 --regression --print-freq 250 "$@"
./train.py --epochs 100 --optimizer Adam --lr 0.001 --deterministic --compress schedule-faceid.yaml --model ai85faceidnetresbn --dataset FaceIDRot90 --batch-size 100 --device MAX78000 --regression --print-freq 250 --use-bias "$@"