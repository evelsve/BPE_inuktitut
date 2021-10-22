#!/bin/bash -l

#SBATCH -A uppmax2021-2-14 # project number for accounting.
#SBATCH -p core -n 4 # resource, 4 cpu cores
#SBATCH -M snowy # cluster name
#SBATCH -t 13:15:00 # time reserved for your job, not the exact time your job will run. If the job takes longer time, you need to increase the time. format: #hour:min:sec#
#SBATCH -J fairseq_BPE10k # job name
#SBATCH --gres=gpu:1 # reserve one GPU for your job
date

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/ik-en \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --no-epoch-checkpoints \
    --stop-time-hours 11
    
date

PREFIX_OUT=checkpoints/generation
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/ik-en \
    --path checkpoints/checkpoint_best.pt --remove-bpe \
    --batch-size 128 --beam 5 | tee $PREFIX_OUT.out

# extract translation and reference from the log
grep '^[T]-' $PREFIX_OUT.out | cut -f2 > $PREFIX_OUT.ref
grep '^[D]-' $PREFIX_OUT.out | cut -f3 > $PREFIX_OUT.trans

# compute the bleu score
fairseq-score -s $PREFIX_OUT.trans -r $PREFIX_OUT.ref 

date