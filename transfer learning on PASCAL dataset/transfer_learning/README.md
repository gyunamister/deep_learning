## Transfer learning implementation

------

Transfer learning on PASCAL multi-class image recognition task.
For pre-trained model, I used inceptionv3-tf.model.
For detailed information, please refer to 'transfer_learning_on_PASCAL.pdf'

------

* you need to download PASCAL image sets into ./VOC2007_test/ and ./VOC2007_train/

### library 설치
pip install -r requirements.txt

### train:
python train.py --image_dir 'path_to_VOC2007'

### predict:
python predict.py --test_dir 'path_to_VOC2007/JPEGImages' --model cust_inceptionv3.model

### evaluate:
python eval.py --test_dir "path_to_VOC2007/ImageSets/Main"