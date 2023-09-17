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
--cgm-args "{'n_encoders': 5000, 'feat_init': 'randomChoice', 'feat_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1}";
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
--cgm-args "{'n_encoders': 3000, 'feat_init': 'randomChoice', 'feat_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1}";
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
--cgm-args "{'n_encoders': 3000, 'feat_init': 'randomChoice', 'feat_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1}";
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
--cgm-args "{'n_encoders': 3000, 'feat_init': 'randomChoice', 'feat_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1}";
done