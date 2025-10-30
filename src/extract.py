"""
main.py

This is the entry point for the project. It orchestrates the training pipeline,
including data preparation, model training, and evaluation.


"""
import os.path
import torch
import wandb
from parameters import params
from models.SELDnet import SELDModel
from losses.loss import SELDLossADPIT, SELDLossSingleACCDOA
from metrics.metrics import ComputeSELDResults
from data.data_generator import DataGenerator
from torch.utils.data import DataLoader
from data.extract_features import SELDFeatureExtractor
import utils.utils as utils
from rich.progress import Progress
import platform
import time
import augment.training_utils as tu
import argparse
import numpy as np
import os
from data.cal_dist_norm import calculate_distance_normalization_params
from models.ResNetConformer import ResNetConformer
def setup_model_and_loss(in_feat_shape, device):
    """
    Initializes the model, loss, and metrics.
    """
    model_name = params['model'].lower() # You can add additional models if you want
    if "seldnet" in model_name:
        model = SELDModel(params=params, in_feat_shape=in_feat_shape).to(device)
        print(f"Using SELDNet!")
    elif "resnetconformer" in model_name:
        model = ResNetConformer(params=params, in_feat_shape=in_feat_shape).to(device)
        print(f"Using ResNetConformer!")
    else:
        raise ValueError("Unknown model type specified.")

    import torchinfo
    model_profile = torchinfo.summary(model, input_size=in_feat_shape)
    print('MACC:\t \t %.3f' %  (model_profile.total_mult_adds/1e9), 'G')
    print('Memory:\t \t %.3f' %  (model_profile.total_params/1e6), 'M\n')

    # # Checking for multiple GPUs
    # n_gpus = torch.cuda.device_count()
    # if n_gpus > 1:
    #     print(f"Found {n_gpus} GPU(s)")
    #     model = torch.nn.DataParallel(model)  # replicates module across GPUs
    #     params['nb_workers'] = int(params['nb_workers'] * n_gpus)
    #     print(f"Using nn.DataParallel on {n_gpus} GPUs, using {params['nb_workers']} workers")

    # Instantiate loss based on the configuration.
    if params['multiACCDOA']:
        loss_fn = SELDLossADPIT(params=params).to(device)
    else:
        loss_fn = SELDLossSingleACCDOA(params=params).to(device)
    metrics = ComputeSELDResults(params=params, ref_files_folder=os.path.join(params['root_dir'], 'metadata_dev'))
    print("Metrics are set up!")
    return model, loss_fn, metrics


def train_epoch(seld_model, dev_train_iterator, optimizer, seld_loss, step_scheduler=None):

    seld_model.train()
    batch_losses = []
    num_batches = len(dev_train_iterator)

    bs = params['batch_size']
    accum_bs = params.get('accum_batch', bs)
    if accum_bs % bs != 0: raise ValueError(f"accum_batch ({accum_bs}) must be a multiple of batch_size ({bs})")
    accum_steps = accum_bs // bs

    optimizer.zero_grad()
    with Progress(transient=True) as progress:
        task = progress.add_task("[red]Training:\t", total=num_batches)

        for batch_idx, (input_features, labels) in enumerate(dev_train_iterator):

            input_features, labels = input_features.to(device), labels.to(device)

            logits = seld_model(input_features)

            # compute loss and scale down by accum_steps
            loss = seld_loss(logits, labels) / accum_steps
            loss.backward()

            true_loss = loss.item() * accum_steps
            batch_losses.append(true_loss)

            # every accum_steps mini‑batches, do optimizer step + scheduler
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
                optimizer.step()
                if step_scheduler is not None: step_scheduler.step()
                optimizer.zero_grad()

            progress.update(task, advance=1)

    # Return average loss
    return float(np.mean(batch_losses))


def val_epoch(seld_model, dev_test_iterator, seld_loss, seld_metrics, output_dir, is_jackknife=False):# , save_predictions=True

    seld_model.eval()
    val_loss_per_epoch = 0  # Track loss per iteration to average over the epoch.
    # # 收集所有预测结果，而不是立即写文件
    # all_logits = []
    # all_label_files = []
    # write_time = 0
    # inference_time = 0
    with Progress(transient=True) as progress:
        task = progress.add_task("[green]Validation:\t", total=len(dev_test_iterator))
        with torch.no_grad():
            
            for j, (input_features, labels) in enumerate(dev_test_iterator):
                input_features, labels = input_features.to(device), labels.to(device)
                # inf_start = time.time()
                # Forward pass
                logits = seld_model(input_features)

                # Compute loss
                loss = seld_loss(logits, labels)
                # inference_time += time.time() - inf_start
                val_loss_per_epoch += loss.item()

                # save predictions to csv files for metric calculations
                start = j * params['val_batch_size']
                end = start + logits.size(0)
                # write_start = time.time()
                utils.write_logits_to_dcase_format(logits, params, output_dir, dev_test_iterator.dataset.label_files[start: end])
                # write_time = write_time + time.time() - write_start
                progress.update(task, advance=1)
                # # 收集预测结果（移到CPU以节省GPU内存）
                # start = j * params['val_batch_size']
                # end = start + logits.size(0)
                # all_logits.append(logits.cpu())
                # all_label_files.extend(dev_test_iterator.dataset.label_files[start:end])
                
                # progress.update(task, advance=1)
    #  # 批量写入文件（一次性操作）
    # all_logits = torch.cat(all_logits, dim=0)
    # if save_predictions: # 是否保存output
    #     utils.write_logits_to_dcase_format(all_logits, params, output_dir, all_label_files)
    #     metric_scores = seld_metrics.get_SELD_Results(pred_files_path=os.path.join(output_dir, 'dev-test'), is_jackknife=is_jackknife)
    # else:
    #     # 直接从内存计算指标（新方式）
    #     predictions_dict = utils.logits_to_prediction_dict(all_logits, params, all_label_files)
    #     metric_scores = seld_metrics.get_SELD_Results_from_memory(predictions_dict, is_jackknife=is_jackknife)
    # Clear up memory
    del input_features, labels, logits
    torch.cuda.empty_cache()
    # metric_start = time.time()
    avg_val_loss = val_loss_per_epoch / len(dev_test_iterator)
    metric_scores = seld_metrics.get_SELD_Results(pred_files_path=os.path.join(output_dir, 'dev-test'), is_jackknife=is_jackknife)
    # metric_time = time.time() - metric_start
    # print(f"  Metrics:   {metric_time:.1f}s ")
    # print(f"  File I/O:  {write_time:.1f}s ")
    # print(f"  Inference: {inference_time:.1f}s ")
    return avg_val_loss, metric_scores

import random
def set_seed(seed):
    """
    设置所有随机种子以确保可复现性
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 确保 CUDA 操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    print(f"Random seed set to {seed}")

def seed_worker(worker_id):
    # 让每个 worker 的 numpy/random 拥有独立且可复现的状态
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(device, args):
    
    normalization_params = calculate_distance_normalization_params(params['root_dir'])
    params.update(normalization_params)
    print(f"Distance normalization parameters added to config: {normalization_params}")
    # Convert and update the params dictionary
    arg_dict = {k: v for k,v in vars(args).items() if v is not None}
    params.update(arg_dict)
    set_seed(params['seed'])  # 设置随机种子
    params['net_type'] = args.exp # Get unique experiment name
    print("Experiment Parameters:")
    for key, value in params.items():
        print(f"\t{key}: {value}")

    # Adjust number of workers based on the system.
    if platform.system() == 'Linux': 
        params['nb_workers'] = 4

    # # Set up directories for storing model checkpoints, predictions(output_dir), and create a summary writer
    # checkpoints_folder, output_dir, summary_writer, reference = utils.setup(params)
    # print(f"Saving best model in: {checkpoints_folder}")

    # Feature extraction code.
    feat_folder = (f"mel{params['nb_mels']}" if params['nb_mels'] else "linspec") + ("_gamma" if params['gamma'] else "") + \
                  ("_ipd" if params['ipd'] else "") + ("_iv" if params['iv'] else "") + ("_slite" if params['slite'] else "") + \
                  ("_ms" if params['ms'] else "") + ("_ild" if params['ild'] else "")  + ("_ildnorm" if params['ildnorm'] else "") + \
                  ("_gcc" if params['gcc'] else "") + ("_dnorm" if params['dnorm'] else "")
    params['feat_dir'] = os.path.join(params['root_dir'], feat_folder)
    print(f"Feature directory: {params['feat_dir']}")
    # Set up feature extractor and preprocessing
    feature_extractor = SELDFeatureExtractor(params)
    feature_extractor.extract_features(split='dev')
    feature_extractor.extract_labels(split='dev')
    feature_extractor.preprocess_features(split='dev')

    # # Setup augmentations if enabled.
    # n_aug_channels = 4 if params['ms'] else 2
    # if params['cutout'] or params['itfm']:
    #     transforms = tu.CompositeCutoutTorch(always_apply=False, p=0.5, aug_channels=n_aug_channels, use_itfm=params['itfm'])
    # elif params['freqshift']:
    #     transforms = tu.RandomShiftUpDownTorch(always_apply=False, p=0.5, aug_channels=n_aug_channels)
    # elif params['filtaug']:
    #     transforms = tu.FilterAugmentNormalized(always_apply=False, p=0.5, aug_channels=n_aug_channels)
    # elif params['compfreq']:
    #     transforms = tu.CompositeFrequencyTorch(always_apply=False, p=0.5, aug_channels=n_aug_channels)
    # else:
    #     transforms = None

    # # Set up dev_train and dev_test data iterator
    # g = torch.Generator()
    # g.manual_seed(params['seed'])
    # dev_train_dataset = DataGenerator(params=params, mode='dev_train', transform=transforms)
    # dev_train_iterator = DataLoader(dataset=dev_train_dataset, batch_size=params['batch_size'], num_workers=params['nb_workers'], shuffle=params['shuffle'],
    #                                 drop_last=False, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    # dev_test_dataset = DataGenerator(params=params, mode='dev_test')
    # dev_test_iterator = DataLoader(dataset=dev_test_dataset, batch_size=params['val_batch_size'], num_workers=params['nb_workers'], shuffle=False, 
    #                                drop_last=False, worker_init_fn=seed_worker, generator=g)

    # # 查看数据集大小
    # train_dataset_size = len(dev_train_dataset)
    # val_dataset_size = len(dev_test_dataset)
    # print(f"Training dataset size: {train_dataset_size}")
    # print(f"Validation dataset size: {val_dataset_size}")

    # # Getting the input feature shape
    # first_batch = next(iter(dev_train_iterator))
    # in_feat_shape, out_feat_shape = first_batch[0].shape, first_batch[1].shape
    # print(f"Number of batches: {len(dev_train_iterator)}\nIn Shape: {in_feat_shape}\tOut Shape: {out_feat_shape}")

    # # Start Training Loops
    # start_epoch, best_epoch = 0, 0
    # best_f_score = float('-inf')
    # best_seld_err = 1.0

    # # Set up model, loss, and metrics
    # seld_model, seld_loss, seld_metrics = setup_model_and_loss(in_feat_shape=in_feat_shape, device=device)

    # # Load pretrained model if specified
    # if args.pretrained_exp:
    #     pretrained_path = os.path.join("checkpoints", args.pretrained_exp, "best_model.pth")
    #     if os.path.isfile(pretrained_path):
    #         print(f"Loading pretrained model from: {pretrained_path}")
    #         pretrained_ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    #         seld_model.load_state_dict(pretrained_ckpt['seld_model'])
            
    #         if args.resume:
    #             start_epoch = pretrained_ckpt.get('epoch', 0) + 1
    #             best_f_score = pretrained_ckpt.get('best_f_score', float('-inf'))
    #             best_seld_err = pretrained_ckpt.get('best_seld_err', 1.0)
    #             print(f"Resuming training from epoch {start_epoch}")
    #             print(f"Previous best F1: {best_f_score * 100:.2f}%, SELD Error: {best_seld_err:.4f}")
    #         else:
    #             print("Loading weights only, starting fresh training from epoch 0")
    #     else:
    #         print(f"Pretrained model not found at: {pretrained_path}")
    #         print("Starting training from scratch...")

    # # Set up optimizer and scheduler
    # if params['weight_decay'] != 0:
    #     no_decay = ["bias", "bn.weight", "bn.bias", "ln.weight", "ln.bias"]
    #     decay_params = []
    #     no_decay_params = []
    #     for name, param in seld_model.named_parameters():
    #         if not param.requires_grad:
    #             continue
    #         if any(nd in name.lower() for nd in no_decay):
    #             no_decay_params.append(param)
    #         else:
    #             decay_params.append(param)
    #     optimizer_grouped_parameters = [{'params': decay_params,    'weight_decay': params['weight_decay']},
    #                                     {'params': no_decay_params, 'weight_decay': 0.0}]        
    #     optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=params['learning_rate'])
    #     print(f"Using Adam with Weight Decay optimizer (Peak LR: {params['learning_rate']}, WD: {params['weight_decay']})")
    # else:
    #     optimizer = torch.optim.Adam(params=seld_model.parameters(), lr=params['learning_rate'])
    #     print(f"Using vanilla Adam optimizer (Peak LR: {params['learning_rate']})")

    # # Now we set the learning rate scheduler
    # total_steps = params['nb_epochs'] * np.ceil(len(dev_train_iterator) / (int(params['accum_batch'] // params['batch_size'])))
    # step_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=params['learning_rate'], total_steps=int(total_steps))
    # print(f"OneCycleLR scheduler used (Total Number of Steps: {total_steps})")
    # print(f"Distance Normalized: {params['dnorm']}\nITFM: {params['itfm']}\n"
    #       f"Cutout: {params['cutout']}\nFreqshift: {params['freqshift']}\n"
    #       f"FiltAug: {params['filtaug']}\nCompFreq: {params['compfreq']}\nAllAug: {params['alltrans']}\n")

    # # # Setup the W&B run
    # os.environ["WANDB_MODE"]="offline"
    # wandb_dir = os.path.join(params['wandb_dir'], reference)
    # os.makedirs(wandb_dir, exist_ok=True)
    # wandb.init(project=params['project'], name=params['exp'], config=params, reinit=True, mode="offline",dir=wandb_dir)
    # wandb.watch(seld_model, log='all', log_freq=100)

    # try:
    #     for epoch in range(start_epoch, params['nb_epochs']):
    #         # ------------- Training -------------- #
    #         train_start_time = time.time()
    #         avg_train_loss = train_epoch(seld_model=seld_model, dev_train_iterator=dev_train_iterator, optimizer=optimizer, seld_loss=seld_loss, step_scheduler=step_scheduler)
    #         train_time = time.time() - train_start_time

    #         # Log training loss to W&B
    #         wandb.log({"Loss/Train": avg_train_loss, "LearningRate": optimizer.param_groups[0]['lr']}, step=epoch)

    #         # -------------  Validation -------------- #
    #         # Perform validation every `val_freq` epochs
    #         is_regular_validation_time = (epoch % params['val_freq'] == 0)

    #         # Always validate every epoch after 80% of total epochs
    #         is_final_phase = (epoch / params['nb_epochs']) > 0.8

    #         # If `full`, we validate each epoch
    #         is_full_training_midway = params.get('full', False)

    #         should_validate = (is_regular_validation_time or is_final_phase or is_full_training_midway)
    #         if should_validate:
    #             val_start_time = time.time()

    #             # # 决定何时保存预测文件
    #             # save_files = (
    #             #     epoch >= params['nb_epochs'] - 1 #or  # 最后5个epoch总是保存
    #             #     # epoch % (params['val_freq'] * 5) == 0 #or  # 每隔一段时间保存
    #             #     # (epoch / params['nb_epochs']) > 0.8  # 训练后期总是保存
    #             # )

    #             # avg_val_loss, metric_scores = val_epoch(seld_model, dev_test_iterator, seld_loss, seld_metrics, output_dir,save_predictions=save_files)
    #             avg_val_loss, metric_scores = val_epoch(seld_model, dev_test_iterator, seld_loss, seld_metrics, output_dir)
    #             val_f, val_ang_error, val_dist_error, val_rel_dist_error, val_onscreen_acc, class_wise_scr = metric_scores
    #             val_seld_error = ((1 - val_f) + (val_ang_error / 180) + val_rel_dist_error)/3 
    #             val_time = time.time() - val_start_time

    #             # Log validation loss and key metrics
    #             wandb.log({"Loss/Validation": avg_val_loss,
    #                        "Metric/F1_Score": val_f,
    #                        "Metric/Angular_Error": val_ang_error,
    #                        "Metric/Distance_Error": val_dist_error,
    #                        "Metric/Relative_Distance_Error": val_rel_dist_error,
    #                        "Metric/Val_3D_SELD_Error": val_seld_error}, step=epoch)

    #         # ------------- Save model if validation f score improves -------------#
    #         # indicator = (val_seld_error <= best_seld_err) if params['finetune'] else (should_validate and val_f >= best_f_score)  
    #         indicator = (should_validate and val_f >= best_f_score)  
    #         if indicator:
    #             best_f_score = val_f
    #             best_epoch = epoch
    #             best_seld_err = val_seld_error
    #             net_save = {'seld_model': seld_model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch, 'params': params,
    #                         'best_f_score': best_f_score, 'best_ang_err': val_ang_error, 'best_rel_dist_err': val_rel_dist_error}
    #             torch.save(net_save, checkpoints_folder + "/best_model.pth")

    #         # ------------- Log losses and metrics ------------- #
    #         print(
    #             f"Epoch {epoch + 1}/{params['nb_epochs']} | "
    #             f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
    #             f"Time: {train_time:.0f}/{val_time:.0f} | "
    #             f"Loss: {avg_train_loss:.4f} | "
    #             f"F1: {val_f * 100:.3f} | "
    #             f"LE: {val_ang_error:.2f} | "
    #             f"DE: {val_dist_error:.2f} | "
    #             f"RDE: {val_rel_dist_error:.3f} | "
    #             f"SELD Error: {val_seld_error:.3f} | "
    #             f"Best: {best_epoch + 1} ({best_f_score * 100:.2f})"
    #         )

    # except KeyboardInterrupt:
    #     print("Training ended prematurely. Calculating results now.")

    # # Evaluate the best model on dev-test.
    # print(f"Loading best model checkpoint from: {os.path.join(checkpoints_folder, 'best_model.pth')}\n\tBest Epoch: {best_epoch + 1}\tBest F-score: {best_f_score * 100:.2f}")
    # best_model_ckpt = torch.load(os.path.join(checkpoints_folder, 'best_model.pth'), map_location=device, weights_only=False)
    # seld_model.load_state_dict(best_model_ckpt['seld_model'])
    # use_jackknife = params['use_jackknife']
    # test_loss, test_metric_scores = val_epoch(seld_model, dev_test_iterator, seld_loss, seld_metrics, output_dir, is_jackknife=use_jackknife)
    # test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr = test_metric_scores
    # utils.print_results(test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr, params)

    # # Close the W&B instance
    # wandb.finish()


if __name__ == '__main__':

    # Record the start time
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    parser = argparse.ArgumentParser(description='DCASE 2025 Task 3 argument parser')

    # Experiment Arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--project', type=str, default='Task3', help='W&B Project Name')
    parser.add_argument('--exp', type=str, default='Baseline', help='Name of the experiment')
    # parser.add_argument('--model', type=str, default='seldnet', help='Model to use')
    # parser.add_argument('--batch_size', type=int, default=64, help='Batch size to use')
    # parser.add_argument('--accum_batch', type=int, default=64, help='True batch size to use')
    # parser.add_argument('--val_batch', type=int, default=64, help='Batch size for the validation dataloader')
    parser.add_argument('--nb_epochs', type=int, default=100, help='Total number of epochs')
    # parser.add_argument('--full', action='store_true', help='If want to validate each epoch')

    # # Learning Arguments
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    # parser.add_argument('--scheduler', type=str, default="one")
    parser.add_argument('--val_freq', type=int, default=2)
    # parser.add_argument('--dropout', type=float, default=0.05)

    # Dataset and Augmentations
    parser.add_argument('--cutout', action='store_true', help='Use the Random Cutout (Time-Frequency Masking) augmentation')
    parser.add_argument('--freqshift', action='store_true', help='Use the Frequency Shifting augmentation')
    parser.add_argument('--filtaug', action='store_true', help='Use the FilterAugment augmentation')
    parser.add_argument('--compfreq', action='store_true', help='Use both Frequency Shifting and FilterAugment augmentation')
    parser.add_argument('--alltrans', action='store_true', help='Use the all the above augmentation')
    parser.add_argument('--itfm', action='store_true', help='Use the Inter-channel Level-Aware TFM variation')

    # Features and Labels
    parser.add_argument('--ms', action='store_true', help='Mid-Side spectrogram')
    parser.add_argument('--gamma', action='store_true', help='Magnitude Squared Coherence')
    parser.add_argument('--ipd', action='store_true', help='Inter-channel Phase Differences')
    parser.add_argument('--iv', action='store_true', help='Mid-Side Intensity Vector')
    parser.add_argument('--slite', action='store_true', help='SALSA-Lite (Normalized Inter-channel Phase Differences)')
    parser.add_argument('--ild', action='store_true', help='Inter-channel Level Differences')
    parser.add_argument('--ildnorm', action='store_true', help='Inter-channel Level Differences Normalization')
    parser.add_argument('--gcc', action='store_true', help='Generalized Cross-Correlation')
    parser.add_argument('--dnorm', action='store_true', help='Distance Normalization')
    parser.add_argument('--multiACCDOA', action='store_false')
    parser.set_defaults(dnorm=False, multiACCDOA=True) # Default always use DNorm and Multi-ACCDOA format

    # Finetuning options
    parser.add_argument('--pretrained_exp', type=str, default=None, help='Name of pretrained experiment to load from')
    parser.add_argument('--resume', action='store_true', help='Resume training from pretrained model')    
    parser.add_argument('--finetune', action='store_true', help='If finetuning, use SELD error as indicator, else use F-score')
    args = parser.parse_args()

    import sys
    try:
        sys.exit(main(device, args))
    except (ValueError, IOError) as e:
        # Handle exceptions and exit with the error
        sys.exit(e)
    finally:
        # Record the end time
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        elapsed_time_str = utils.format_elapsed_time(elapsed_time)
        print(f"Execution time: {elapsed_time_str}")