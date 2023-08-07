import os

import torch
from tqdm import tqdm
import wandb
from utils.utils import get_lr
        
def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0

    task_name = model.task_name
    task_train_loss = [0 for i in task_name]
    task_train_num = [0 for i in task_name]
    task_val_loss = [0 for i in task_name]
    task_val_num = [0 for i in task_name]



    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break


        optimizer.zero_grad()
        loss_value = 0

        for tag_temp,(images, bboxes) in enumerate(batch):
            if len(images) == 0:
                continue
            with torch.no_grad():
                if cuda:
                    images = images.cuda(local_rank)
                    bboxes = bboxes.cuda(local_rank)

            if not fp16:
                outputs = model_train(images)
                temp_loss = yolo_loss(outputs[tag_temp], bboxes,tag_temp)
                loss_value += temp_loss
                task_train_loss[tag_temp] += temp_loss.item()//len(images)
                task_train_num[tag_temp] += 1

            else:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs         = model_train(images)
                    temp_loss = yolo_loss(outputs[tag_temp], bboxes,tag_temp)
                    loss_value += temp_loss
                    task_train_loss[tag_temp] += temp_loss.item() // len(images)
                    task_train_num[tag_temp] += 1

        if not fp16:
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            optimizer.step()
        else:
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)


    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        optimizer.zero_grad()
        loss_value = 0
        for tag_temp,(images, bboxes) in enumerate(batch):
            if len(images) == 0:
                continue
            with torch.no_grad():
                if cuda:
                    images = images.cuda(local_rank)
                    bboxes = bboxes.cuda(local_rank)
                #----------------------#
                #   清零梯度
                #----------------------#
                # optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                outputs     = model_train_eval(images)
                temp_loss = yolo_loss(outputs[tag_temp], bboxes,tag_temp)
                loss_value  += temp_loss

                task_val_loss[tag_temp] += temp_loss.item()//len(images)
                task_val_num[tag_temp] += 1
        val_loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        wandb_log_write = {"All/Train Loss": loss / (iteration + 1),"All/Val Loss": val_loss / (iteration + 1),"Learning Rate": get_lr(optimizer)}
        for tag_temp in range(len(task_train_loss)):
            task_name_temp = task_name[tag_temp]
            wandb_log_write['Train_loss/'+task_name_temp] = task_train_loss[tag_temp]/task_train_num[tag_temp]
            wandb_log_write['Val_loss/'+task_name_temp] = task_val_loss[tag_temp]/task_val_num[tag_temp]



    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        map_return = eval_callback.on_epoch_end(epoch + 1, model_train_eval)

        wandb_log_write.update(map_return)
        wandb.log(wandb_log_write)

        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
