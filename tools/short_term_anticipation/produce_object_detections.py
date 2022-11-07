"""This script can be used to extract object detections from the annotated frames"""
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pickle
import json

from detectron2.config import get_cfg
from os.path import join
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import cv2
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path_to_checkpoint",
        required=False,
        type=Path,
        default="/local/home/rpasca/Thesis/Ego4d_all/forecast/short_term_anticipation/models/object_detector.pth",
    )
    parser.add_argument(
        "--path_to_sta_annotations", required=False, type=Path, default="/data/rpasca/Datasets/Ego4d/data/annotations/"
    )
    parser.add_argument(
        "--path_to_images",
        required=False,
        type=Path,
        default="/local/home/rpasca/Thesis/Ego4d_all/forecast/tools/short_term_anticipation/data/object_frames/",
    )
    parser.add_argument("--path_to_output_json", required=False, type=Path, default="redo_their_rgb_preds.json")

    args = parser.parse_args()

    train = json.load(open(args.path_to_sta_annotations / "fho_sta_train.json"))
    val = json.load(open(args.path_to_sta_annotations / "fho_sta_val.json"))
    # test = json.load(open(args.path_to_sta_annotations / "fho_sta_test_unannotated.json"))

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = str(args.path_to_checkpoint)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(train["noun_categories"])

    predictor = DefaultPredictor(cfg)

    detections = {}

    print(cfg)

    # for anns in [train, val, test]:
    for anns in [val]:
        print(type(predictor))

        for ann in tqdm(anns["annotations"]):
            uid = ann["uid"]
            name = f"{uid}.jpg"
            img_path = args.path_to_images / name
            img = cv2.imread(str(img_path))
            outputs = predictor(img)["instances"].to("cpu")

            dets = []
            for box, score, noun in zip(
                # outputs.pred_boxes.tensor, outputs.scores, outputs.pred_classes, outputs.all_scores
                outputs.pred_boxes.tensor,
                outputs.scores,
                outputs.pred_classes,
            ):
                box = box.tolist()
                box = [float(x) for x in box]
                score = score.item()
                noun = noun.item()
                # all_scores = [float(x) for x in all_scores]
                # dets.append({"box": box, "score": score, "noun_category_id": noun, "all_scores": all_scores})
                dets.append({"box": box, "score": score, "noun_category_id": noun})
            detections[uid] = dets

    json.dump(detections, open(args.path_to_output_json, "w"), indent=3)
