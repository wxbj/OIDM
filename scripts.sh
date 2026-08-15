# ==========================================
# 1. ACDC 数据集 (2D UNet)
# ==========================================
# 训练
python ACDC_train.py -cfg configs/UNet_ACDC.yaml
# 测试
python test_ACDC.py -cfg configs/UNet_ACDC.yaml

# ==========================================
# 2. LA 数据集 (3D VNet)
# ==========================================
# 训练
python LA_train.py -cfg configs/VNet_LA.yaml
# 测试
python test_LA.py -cfg configs/VNet_LA.yaml

# ==========================================
# 3. promise12 数据集 (2D UNet)
# ==========================================
# 训练
python Promise12_train.py -cfg configs/UNet_promise12.yaml
# 测试
python test_promise12.py -cfg configs/UNet_promise12.yaml

# ==========================================
# 4. ISIC-2017 数据集 (2D UNet)
# ==========================================
# 训练
python ISIC_train.py -cfg configs/UNet_ISIC.yaml
# 测试
python test_ISIC.py -cfg configs/UNet_ISIC.yaml
