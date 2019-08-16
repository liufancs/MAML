# User Diverse Preferences Modeling By Multimodal Attentive Metric Learning 

This is our implementation for the paper:

Fan Liu, Zhiyong Cheng*, Changchang, Yinglong Wang, Liqiang Nie*, Mohan Kankanhalli. User Diverse Preference Modeling via Multimodal Attentive Metric Learning. ACM International Conference on Multimedia (MM'19), Nice, France, 2019 (“*”= Corresponding author:)

**Please cite our ACMMM'19 paper if you use our codes. Thanks!**

## Environment Settings
-Tensorflow-gpu version: 1.3.0

## Example to run the codes.

Run MAML.py
```
python MAML.py --dataset Office --num_neg 4 --eva_batches 400 --batch_size 5000 --hidden_layer_dim 256 --margin 1.6 --dropout 0.2 --feature_l2_reg 5.0
```

### Dataset
We provide four processed datasets: Amazon-Office, Amazon-MenClothing, Amazon-WomenClothing, Amazon-Toys&Games.
train_csv:
- Train file.

test_csv:
- Test file.

asin_sample.json:
- Negative instances of items.

imge_feature.npy:
- Image features.

doc2vecFile:
- Text features.

All of the above files could be downloaded from :
- Link: https://pan.baidu.com/s/1-95zBrQxx-pSmTV-MS5cBQ  
- Extract code: appe 

Last Update Date: AUG 16, 2019
