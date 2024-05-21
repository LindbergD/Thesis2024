import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from models.loss import NTXentLoss


def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    # To study losses over self-supervised training
    if training_mode == "self_supervised":
        nt_xent_losses = torch.empty(size=(config.num_epoch, len(train_dl))).to(device) # size=(num_epochs, num_batches)
        nce_losses = torch.empty(size=(config.num_epoch, len(train_dl))).to(device)
        # Stores the loss of every batch in each epoch


    # Loop through each epoch, training and evaluating the model
    for epoch in range(1, config.num_epoch + 1): 
        # Train
        train_loss, train_acc, nt_xent_loss, nce_loss = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_dl, config, device, training_mode)

        if training_mode == "self_supervised":
            # Add losses to analyze afterwards
            nt_xent_losses[epoch-1] = nt_xent_loss
            nce_losses[epoch-1] = nce_loss

            # Don't evaluate on validation data, only log train loss and accuracy
            logger.debug(f'\nEpoch : {epoch}\n'
                        f'Train Loss     : {train_loss:.4f}')
            
        elif training_mode != 'self_supervised': 
            # Calculate loss and accuracy on validation set
            valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)

            scheduler.step(valid_loss) # Use scheduler (in all other modes but self-supervised)

            logger.debug(f'\nEpoch : {epoch}\n'
                        f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                        f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

    logger.debug(f"\n\nFinished All Epochs \t|\t Saving Model .... \n")

    # Save the model parameters
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
    
    # Save losses
    if training_mode == "self_supervised":
        os.makedirs(os.path.join(experiment_log_dir, "losses"), exist_ok=True)
        save_path_nt_xent = os.path.join(experiment_log_dir, "losses", "nt_xent_losses.pt")
        save_path_nce = os.path.join(experiment_log_dir, "losses", "nce_losses.pt")

        torch.save(nt_xent_losses, save_path_nt_xent)
        torch.save(nce_losses, save_path_nce)

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config, device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    # To study the loss values over training
    nt_xent_losses = []
    nce_losses = []

    for batch_idx, (source, target, aug) in enumerate(train_loader):

        source, target = source.float().to(device), target.long().to(device)
        # >> source (128, 1, 3000), target (128)
        aug = aug.float().to(device)
        # >> (128, 1, 3000)

        # zero out optimizer's
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        # Goes over the training process, only relevant in self-supervised mode
        if training_mode == "self_supervised":

            # Obtain class predictions and latent space embeddings
            predictions_source, features_source = model(source)
            predictions_aug, features_aug = model(aug)
            # >> features are (128, 128, 127) = (batch, output_channels, features_len)
            # >> predictions are (128, 5) = (batch, output_classes)
            # At this point, the original 128 recordings in this batch have been squeezed into a latent representations (128, 127)

            # batch normalization, normalizes over the channel dimensions for each batch
            features_source = F.normalize(features_source, dim=1)
            features_aug = F.normalize(features_aug, dim=1)
            # >> Ensures that feature vectors are unit vectors

            # Run the latent space prediction task with the TC model
            tc_loss, tc_c_projection_source, tc_c_projection_aug = temporal_contr_model(features_source, features_aug)
                                    
            # normalize projection feature vectors
            zis = tc_c_projection_source 
            zjs = tc_c_projection_aug 
            # >> (128, 32) each batch encoded as 32 features, after non-linear projection head

            # Compute loss
            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)            
            
            nt_xent_loss = nt_xent_criterion(zis, zjs)
            loss = tc_loss * lambda1 +  nt_xent_loss * lambda2

            # To study losses over training
            nt_xent_losses.append(nt_xent_loss.item())
            nce_losses.append(tc_loss.item())

        # Training mode: train_linear, fine_tune, or supervised
        else: 
            # Get class predictions from model
            predictions, _ = model(source)
            
            # Compute loss
            loss = criterion(predictions, target)
            total_acc.append(target.eq(predictions.detach().argmax(dim=1)).float().mean())

        # Store loss
        total_loss.append(loss.item())

        # Compute backprop and update parameters
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    # Average loss for this epoch
    total_loss = torch.tensor(total_loss).mean()
    if training_mode=="self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()

    # To study losses over training
    nt_xent_losses = torch.tensor(nt_xent_losses).to(device)
    nce_losses = torch.tensor(nce_losses).to(device)
    
    return total_loss, total_acc, nt_xent_losses, nce_losses


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for source, target, _ in test_dl:
            source, target = source.float().to(device), target.long().to(device)
            
            predictions, features = model(source)
            loss = criterion(predictions, target)
            total_acc.append(target.eq(predictions.detach().argmax(dim=1)).float().mean())
            total_loss.append(loss.item())

            pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, target.data.cpu().numpy())

    total_loss = torch.tensor(total_loss).mean()  # average loss
    total_acc = torch.tensor(total_acc).mean()  # average acc
    
    return total_loss, total_acc, outs, trgs
