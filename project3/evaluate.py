from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import skimage.io as io
import pylab

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

annFile = '/remote-home/jianghaitian/NNDLPJ3/annotations_DCC/captions_split_set_bottle_val_val_novel2014.json'
resFile = '/remote-home/jianghaitian/NNDLPJ3/CLIP_prefix_caption/bottle.json'
coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

cocoEval = COCOEvalCap(coco, cocoRes, "corpus")
cocoEval.params['image_id'] = cocoRes.getImgIds()
cocoEval.evaluate()

for metric, score in cocoEval.eval.items():
    print(f"{metric:7} {score:.5f}")
