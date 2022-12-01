import os
import pickle
import pprint

import Ego4d_all.forecast.ego4d.utils.logging as logging
import numpy as np
import torch
from Ego4d_all.forecast.ego4d.tasks.short_term_anticipation import ShortTermAnticipationTask
from Ego4d_all.forecast.ego4d.utils.c2_model_loading import get_name_convert_func
from Ego4d_all.forecast.ego4d.utils.parser import load_config, parse_args
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from Ego4d_all.forecast.scripts.slurm import copy_and_run_with_config
from pytorch_lightning.plugins import DDPPlugin

import json
import datetime

logger = logging.get_logger(__name__)


def main(cfg):
    seed_everything(cfg.RNG_SEED)

    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("Run with config:")
    logger.info(pprint.pformat(cfg))

    # Choose task type based on config.
    if cfg.DATA.TASK == "short_term_anticipation":
        TaskType = ShortTermAnticipationTask

    task = TaskType(cfg)

    # Load model from checkpoint if checkpoint file path is given.
    ckp_path = cfg.CHECKPOINT_FILE_PATH
    cfg.DATA.CHECKPOINT_MODULE_FILE_PATH = ""
    if len(ckp_path) > 0:
        if cfg.CHECKPOINT_VERSION == "caffe2":
            with open(ckp_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            state_dict = data["blobs"]
            fun = get_name_convert_func()
            state_dict = {
                fun(k): torch.from_numpy(np.array(v))
                for k, v in state_dict.items()
                if "momentum" not in k and "lr" not in k and "model_iter" not in k
            }

            if not cfg.CHECKPOINT_LOAD_MODEL_HEAD:
                state_dict = {k: v for k, v in state_dict.items() if "head" not in k}
            print(task.model.load_state_dict(state_dict, strict=False))
            print(f"Checkpoint {ckp_path} loaded")
        elif cfg.DATA.CHECKPOINT_MODULE_FILE_PATH != "":

            # Load slowfast weights into backbone submodule
            ckpt = torch.load(
                cfg.DATA.CHECKPOINT_MODULE_FILE_PATH,
                map_location=lambda storage, loc: storage,
            )

            def remove_first_module(key):
                return ".".join(key.split(".")[1:])

            state_dict = {remove_first_module(k): v for k, v in ckpt["state_dict"].items() if "head" not in k}
            missing_keys, unexpected_keys = task.model.backbone.load_state_dict(state_dict, strict=False)
            # Ensure only head key is missing.
            assert len(unexpected_keys) == 0
            assert all(["head" in x for x in missing_keys])
        else:
            # Load all child modules except for "head" if CHECKPOINT_LOAD_MODEL_HEAD is
            # False.
            pretrained = TaskType.load_from_checkpoint(ckp_path)
            state_dict_for_child_module = {
                child_name: child_state_dict.state_dict()
                for child_name, child_state_dict in pretrained.model.named_children()
            }
            print("-----------------------------------------------------")
            print(f"LOADING FROM {ckp_path}")
            print("-----------------------------------------------------")
            for child_name, child_module in task.model.named_children():
                if not cfg.CHECKPOINT_LOAD_MODEL_HEAD and "head" in child_name:
                    continue

                state_dict = state_dict_for_child_module[child_name]
                child_module.load_state_dict(state_dict)

    checkpoint_callback = ModelCheckpoint(monitor="val/ttc_error", mode="min", save_last=True,
                        dirpath=cfg.RESULTS_JSON,
                        filename='{epoch}-{val_loss:.2f}-{val/ttc_error:.2f}',
                        save_top_k=1)
    
    
    if cfg.ENABLE_LOGGING:
        args = {"callbacks": [LearningRateMonitor(), checkpoint_callback]}
    else:
        args = {"logger": False, "callbacks": checkpoint_callback}

    torch.set_num_threads(7)

    trainer = Trainer(
        gpus=[gpu],
        num_nodes=cfg.NUM_SHARDS,
        accelerator=cfg.SOLVER.ACCELERATOR,
        max_epochs=cfg.SOLVER.MAX_EPOCH,
        num_sanity_val_steps=3 if not run_validation_sanity_check else -1,
        benchmark=True,
        log_gpu_memory="min_max",
        replace_sampler_ddp=False,
        fast_dev_run=False,
        default_root_dir=cfg.OUTPUT_DIR,
        plugins=DDPPlugin(find_unused_parameters=False),
        auto_select_gpus=False,
        **args,
    )

    print("---------------------------------------------------")
    print(f"Results will be saved to {cfg.RESULTS_JSON}")
    print("---------------------------------------------------")

    if cfg.TRAIN.ENABLE and cfg.TEST.ENABLE:
        trainer.fit(task)

        # Calling test without the lightning module arg automatically selects the best
        # model during training.
        return trainer.test()

    elif cfg.TRAIN.ENABLE:
        return trainer.fit(task)

    elif cfg.TEST.ENABLE:
        return trainer.test(task)


def get_only_bboxes_obj_detections_path(obj_detection_for_2stage):
    # with open(orig_detections_p, "r") as fp:
    #     orig_dets = json.loads(fp.read())

    with open(obj_detection_for_2stage, "r") as fp:
        obj_dets = json.loads(fp.read())

    # orig_keys = set(orig_dets.keys())
    # our_keys = set(obj_dets["results"].keys())

    # print (f"Missing keys:{len(orig_keys-our_keys)}")
    # if len(orig_keys-our_keys) < 20:
    #     print(orig_keys-our_keys)

    file_name = f'{obj_detection_for_2stage.split(".")[0]}_obj_detections.json'
    print(f"Saving bboxes file for 2 stage to {file_name}")
    with open(file_name, "w") as output:
        json.dump(obj_dets["results"], output, indent=3)

    return file_name

if __name__ == "__main__":
    args = parse_args()
    args.cfg_file = "/local/home/rpasca/Thesis/Ego4d_all/forecast/configs/Ego4dShortTermAnticipation/SLOWFAST_32x1_8x4_R50_no_verb.yaml"

    run_validation_sanity_check = args.run_val

    gpu = args.gpu
    # gpu = 1
    print(f"{gpu=}")
    del args.gpu

    cfg = load_config(args)
    # obj_detections_path = "/local/home/rpasca/Thesis/pred_jsons/corect_true_feather_preds_obj_detections.json"
    # obj_detections_path = "/local/home/rpasca/Thesis/pred_jsons/breezy_sound_2463_test_obj_detections.json"
    # obj_detections_path = "/local/home/rpasca/Thesis/pred_jsons/r_donkey_2603_sharedW_test_obj_detections.json"
    # obj_detections_path = args.obj_detections
    obj_detections_path = "/local/home/rpasca/Thesis/pred_jsons/good_blaze_test_6.json"
    
    obj_detections_path = get_only_bboxes_obj_detections_path(obj_detections_path)
    
    e = datetime.datetime.now()

    if run_validation_sanity_check:
        cfg.TRAIN.ENABLE = True
        cfg.SOLVER.MAX_EPOCH = 0

    str_time = e.strftime("%m-%d_%H:%M")

    cfg.EGO4D_STA.OBJ_DETECTIONS = obj_detections_path
    if cfg.TEST.ENABLE and not cfg.TRAIN.ENABLE:
        cfg.RESULTS_JSON = obj_detections_path.split("/")[-1].split(".")[0]+f"/{str_time}_test_run"
    else:
        cfg.RESULTS_JSON = obj_detections_path.split("/")[-1].split(".")[0]+f"/{str_time}_val_run"
           
    os.makedirs(cfg.RESULTS_JSON, exist_ok=True)
    with open(obj_detections_path, "r") as fp:
        content = json.loads(fp.read())
    
   
    cfg.EGO4D_STA.RGB_LMDB_DIR = os.path.expandvars("$DATA/Ego4d/data/lmdb")
    cfg.EGO4D_STA.ANNOTATION_DIR = os.path.expandvars("$DATA/Ego4d/data/annotations")
    print(cfg)
    if args.on_cluster:
        copy_and_run_with_config(
            main,
            cfg,
            args.working_directory,
            job_name=args.job_name,
            time="72:00:00",
            partition="learnfair",
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=10,
            mem="470GB",
            nodes=cfg.NUM_SHARDS,
            constraint="volta32gb",
        )
    else:  # local
        main(cfg)
