wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P data/
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip -P data/
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip -P data/

unzip data/captions_train-val2014.zip -d ./
unzip data/train2014.zip -d images/
rm data/train2014.zip 
unzip data/val2014.zip -d images/ 
rm data/val2014.zip 
