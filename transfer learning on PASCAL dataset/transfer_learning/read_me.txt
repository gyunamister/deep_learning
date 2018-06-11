#library 설치
pip install -r requirements.txt

train:
python train.py --image_dir 'path_to_VOC2007'

predict:
python predict.py --test_dir 'path_to_VOC2007/JPEGImages' --model cust_inceptionv3.model

evaluate:
python eval.py --test_dir "path_to_VOC2007/ImageSets/Main"
