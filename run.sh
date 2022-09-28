batch=200
group=$1
unseen_classes=oos
detect_method="energylocal"
seed= 38

if [ $group = 0 ];then
    echo "Baseline : Cross-entropy Loss"
    ce_pre_epoches=10
    python main.py --dataset CLINC_OOD_full --mode both --unseen_classes $unseen_classes --metrics_path "./result/ce.csv" --detect_method $detect_method --seed $seed --lmcl --batch_size $batch --cuda --train_class_num n --ce_pre_epoches $ce_pre_epoches --experiment_No ce$ce_pre_epoches

elif [ $group = 1 ];then
    echo "Baseline :  Supervised contrastive loss"
    confused_pre_epoches=0
    global_pre_epoches=10
    ce_pre_epoches=10
    python main.py --dataset CLINC_OOD_full --mode both --unseen_classes $unseen_classes --metrics_path "./result/scl.csv" --detect_method $detect_method --seed $seed --lmcl --batch_size $batch --cuda --train_class_num n --ce_pre_epoches $ce_pre_epoches --confused_pre_epoches 0 --global_pre_epoches $global_pre_epoches --experiment_No ce$ce_pre_epoches-scl$global_pre_epoches --scl_cont

elif [ $group = 2 ];then
    echo "Reassigned Contrastive Learning"
    confused_pre_epoches=5
    global_pre_epoches=10
    ce_pre_epoches=20
    python main.py --dataset CLINC_OOD_full --mode both --unseen_classes $unseen_classes --metrics_path "./result/rcl.csv" --detect_method $detect_method --seed $seed --lmcl --batch_size $batch --cuda --train_class_num n --ce_pre_epoches $ce_pre_epoches --confused_pre_epoches $confused_pre_epoches --global_pre_epoches $global_pre_epoches --experiment_No ce$ce_pre_epoches-rcl$confused_pre_epoches --rcl_cont
fi
