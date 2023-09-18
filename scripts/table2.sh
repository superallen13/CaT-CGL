seed=1024
repeat=5
epoch=500

# CoraFull
dataset=corafull
budget=4

for method in ergnn ssm cgm
do
python train.py \
--seed $seed \
--repeat $repeat \
--cls-epoch $epoch \
--cgl-method $method \
--tim \
--data-dir /scratch/itee/uqyliu71/CaT/data \
--result-path /scratch/itee/uqyliu71/CaT/results \
--dataset-name $dataset \
--budget $budget \
--cgm-args "{'n_encoders': 5000, 'feat_init': 'randomChoice', 'feat_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1, 'activation': True}";
done

# Arxiv
dataset=arxiv
budget=29

for method in ergnn ssm cgm
do
python train.py \
--seed $seed \
--repeat $repeat \
--cls-epoch $epoch \
--cgl-method $method \
--tim \
--data-dir /scratch/itee/uqyliu71/CaT/data \
--result-path /scratch/itee/uqyliu71/CaT/results \
--dataset-name $dataset \
--budget $budget \
--cgm-args "{'n_encoders': 3000, 'feat_init': 'randomChoice', 'feat_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1, 'activation': True}";
done


# Reddit
dataset=reddit
budget=40

for method in ergnn ssm cgm
do
python train.py \
--seed $seed \
--repeat $repeat \
--cls-epoch $epoch \
--cgl-method $method \
--tim \
--data-dir /scratch/itee/uqyliu71/CaT/data \
--result-path /scratch/itee/uqyliu71/CaT/results \
--dataset-name $dataset \
--budget $budget \
--cgm-args "{'n_encoders': 3000, 'feat_init': 'randomChoice', 'feat_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1, 'activation': True}";
done

# Reddit
dataset=products
budget=318

for method in ergnn ssm cgm
do
python train.py \
--seed $seed \
--repeat $repeat \
--cls-epoch $epoch \
--cgl-method $method \
--tim \
--data-dir /scratch/itee/uqyliu71/CaT/data \
--result-path /scratch/itee/uqyliu71/CaT/results \
--dataset-name $dataset \
--budget $budget \
--cgm-args "{'n_encoders': 3000, 'feat_init': 'randomChoice', 'feat_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1, 'activation': True}";
done

seed=1024
repeat=1
epoch=500
dataset=arxiv
budget=29

for method in cgm
do
python train.py \
--seed $seed \
--repeat $repeat \
--cls-epoch $epoch \
--cgl-method $method \
--tim \
--data-dir /scratch/itee/uqyliu71/CaT/data \
--result-path /scratch/itee/uqyliu71/CaT/results \
--dataset-name $dataset \
--budget $budget \
--cgm-args "{'n_encoders': 1000, 'feat_init': 'randomChoice', 'feat_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1, 'activation': True}";
done