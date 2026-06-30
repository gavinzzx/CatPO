import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
import numpy as np
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from .base_todpo_trainer import BaseDPOTrainer

class LlavaDPOTrainer(BaseDPOTrainer):
        
    def concatenated_forward(
        self, model, inputs
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        torch.set_printoptions(profile="full")
        np.set_printoptions(threshold=np.inf)        
        
        
        images = inputs["images"]   
        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_labels = inputs["chosen_labels"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        reject_input_ids = inputs["reject_input_ids"]
        reject_labels = inputs["reject_labels"]
        reject_attention_mask = inputs["reject_attention_mask"]


            
        chosen_weights = inputs["chosen_weights"]  # [batch, seq_lenC]     
        reject_weights = inputs["reject_weights"]  # [batch, seq_lenR]            
        
        # print(f"chosen_weights.shape: {inputs['chosen_weights'].shape}")#torch.Size([1, 13]) 

        # print(f"chosen_input_ids.shape: {chosen_input_ids.shape}")#1 101
        
        # print(f"chosen_input_ids: {chosen_input_ids}")
        
        # print(f"chosen_labels.shape: {chosen_labels.shape}")  #1 101
        # print(f"chosen_labels: {chosen_labels}")      

        # print(f"chosen_weights.shape: {chosen_weights.shape}")#1 17
        # print(f"chosen_weights: {chosen_weights}")
        # print(f"reject_weights.shape: {reject_weights.shape}")
        # print(f"reject_weights: {reject_weights}")
        max_dim = max(chosen_input_ids.shape[1], reject_input_ids.shape[1])
        batch_input_ids = torch.zeros((chosen_input_ids.shape[0]*2, max_dim), dtype=chosen_input_ids.dtype, device=chosen_input_ids.device)
        batch_labels = torch.ones((chosen_input_ids.shape[0]*2, max_dim), dtype=chosen_labels.dtype, device=chosen_labels.device) * -100
        batch_attention_mask = torch.zeros((chosen_input_ids.shape[0]*2, max_dim), device=chosen_attention_mask.device).to(torch.bool)
        
        batch_input_ids[:chosen_input_ids.shape[0], :chosen_input_ids.shape[1]] = chosen_input_ids
        batch_input_ids[reject_input_ids.shape[0]:, :reject_input_ids.shape[1]] = reject_input_ids
        batch_labels[:chosen_labels.shape[0], :chosen_labels.shape[1]] = chosen_labels
        batch_labels[reject_labels.shape[0]:, :reject_labels.shape[1]] = reject_labels
        batch_attention_mask[:chosen_attention_mask.shape[0], :chosen_attention_mask.shape[1]] = chosen_attention_mask
        batch_attention_mask[reject_attention_mask.shape[0]:, :reject_attention_mask.shape[1]] = reject_attention_mask

        # print(f"batch_input_ids.shape--before model pre: {batch_input_ids.shape}")#torch.Size([2, 101]) 2nd data
        # print(f"batch_input_ids--before model pre: {batch_input_ids}")

        # print(f"batch_labels.shape--before model pre: {batch_labels.shape}")   #torch.Size([2, 101])
        # print(f"batch_labels--before model pre: {batch_labels}")        
        

        # print(f"batch_input_ids[0]: {batch_input_ids[0]}")
        # print(f"batch_input_ids[1]: {batch_input_ids[1]}")

        
        # print(f"batch_input_ids.shape: {batch_input_ids.shape}")# print shape of batch_input_ids & batch_labels      torch.Size([2, 96])
        # print(f"batch_labels.shape: {batch_labels.shape}")  #torch.Size([2, 96])


        batch_weights = torch.zeros((chosen_weights.shape[0]*2, batch_labels.shape[1]+575), device=chosen_weights.device, dtype=chosen_weights.dtype)
        #总长度：batch_labels.shape[1]+575              chosen_weights.shape[1]         reject_weights.shape[1]
        if chosen_input_ids.shape[1] > reject_input_ids.shape[1]:
            d_value = chosen_input_ids.shape[1] - reject_input_ids.shape[1] 
            batch_weights[: chosen_weights.shape[0], batch_labels.shape[1]+575-chosen_weights.shape[1]-1:-1] = chosen_weights
            batch_weights[reject_weights.shape[0]:, batch_labels.shape[1]+575-d_value-1-reject_weights.shape[1]:batch_labels.shape[1]+575-d_value-1] = reject_weights
            
        elif chosen_input_ids.shape[1] == reject_input_ids.shape[1]:
            batch_weights[: chosen_weights.shape[0], batch_labels.shape[1]+575-chosen_weights.shape[1]-1:-1] = chosen_weights
            batch_weights[reject_weights.shape[0]:, batch_labels.shape[1]+575-reject_weights.shape[1]-1:-1] = reject_weights
            
        else: #chosen_input_ids.shape[1] < reject_input_ids.shape[1]
            d_value = reject_input_ids.shape[1] - chosen_input_ids.shape[1]
            batch_weights[: chosen_weights.shape[0], batch_labels.shape[1]+575-d_value-1-chosen_weights.shape[1]:batch_labels.shape[1]+575-d_value-1] = chosen_weights
            batch_weights[reject_weights.shape[0]:, batch_labels.shape[1]+575-reject_weights.shape[1]-1:-1] = reject_weights        
        
        
        

        # batch_weights = torch.zeros((chosen_weights.shape[0]*2, batch_labels.shape[1]+575), device=chosen_weights.device, dtype=chosen_weights.dtype)
        # batch_weights[: chosen_weights.shape[0], : chosen_weights.shape[1]] = chosen_weights
        # batch_weights[reject_weights.shape[0]:, : reject_weights.shape[1]] = reject_weights
        # batch_weights[:, reject_weights.shape[1]:batch_labels.shape[1]] = 1    
        
        # batch_weights[:, batch_labels.shape[1]:batch_labels.shape[1]+575] = 0
        # print(f"Before prepare_inputs_labels_for_multimodal:")
        # print(f"batch_input_ids.shape: {batch_input_ids.shape}")
        # print(f"batch_attention_mask.shape: {batch_attention_mask.shape}")
        # print(f"batch_labels.shape: {batch_labels.shape}")
        # print(f"images.shape: {images.shape}")#torch.Size([1, 3, 336, 336])
        
        # prepare inputs        
        result_from_self_model= self.model.prepare_inputs_labels_for_multimodal(
            input_ids=batch_input_ids,
            position_ids=None,
            attention_mask=batch_attention_mask,
            past_key_values=None,
            labels=batch_labels,
            images=torch.cat([images, images], dim=0),
        )
        if result_from_self_model is None:
            raise ValueError("prepare_inputs_labels_for_multimodal returned None!")
        # print(f"result_from_self_model: {result_from_self_model}")
        (
            batch_input_ids,    #None
            batch_position_ids,    #None
            batch_attention_mask,
            batch_past_key_values,    #None
            batch_inputs_embeds,
            batch_labels
        ) = result_from_self_model
        
        # print(f"batch_labels.shape--after model pre: {batch_labels.shape}") #torch.Size([2, 676])
        # print(f"batch_labels--after model pre: {batch_labels}")
        # print(f"Processed batch_input_ids.shape: {batch_input_ids.shape}")
        # print(f"Processed batch_labels.shape: {batch_labels.shape}")
        
        # calculate logits
        all_logits = model.forward(
            inputs_embeds=batch_inputs_embeds,
            labels=None,
            attention_mask=batch_attention_mask,
        ).logits.to(torch.float32)
        cal_batch_logp = self._get_batch_logps
        all_logps = cal_batch_logp(
            all_logits,
            batch_labels,
            average_log_prob=False,
            token_weights=batch_weights, 
        )
        
  
        len_chosen = chosen_input_ids.shape[0]
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        
        
        # don't count image embeds logits
        loss_mask = batch_labels != -100    

        
        logits = [all_logits[i][loss_mask[i]] for i in range(loss_mask.shape[0])]
        chosen_logits = logits[:len_chosen]
        rejected_logits = logits[len_chosen:]
        chosen_logits = [l.detach().cpu().mean() for l in chosen_logits]
        rejected_logits = [l.detach().cpu().mean() for l in rejected_logits]
        chosen_logits = sum(chosen_logits)/len_chosen
        rejected_logits = sum(rejected_logits)/len_chosen

        # Before return 
        # print(f"chosen_logps: {chosen_logps}")
        # print(f"rejected_logps: {rejected_logps}")
        # print(f"chosen_logits: {chosen_logits}")
        # print(f"rejected_logits: {rejected_logits}")        

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, all_logits, batch_labels, batch_weights)  
    
    def get_batch_metrics(
        self,
        inputs,
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}
        
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_logits,        
            batch_labels,        
            batch_weights       
        ) = self.concatenated_forward(self.model, inputs)   
        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
                _,                   
                _,                   
                _                            
            ) = self.concatenated_forward(self.ref_model, inputs)

        policy_rejected_logps = policy_rejected_logps
        reference_rejected_logps = reference_rejected_logps
        

        (policy_chosen_logps, policy_rejected_logps,
     policy_chosen_logits, policy_rejected_logits,
     policy_logits, batch_labels, batch_weights) = \
        self.concatenated_forward(self.model, inputs)

        with torch.no_grad():
            (_, _, _, _,
            ref_logits, _, _) = self.concatenated_forward(self.ref_model, inputs)
        token_kl = self._get_weighted_kl(
        policy_logits, ref_logits, batch_labels, batch_weights
    )

        
        # ======📍📍📍KL--warm up======
        total_steps   = self.args.max_steps
        warm_steps    = int(self.args.lambda_kl_warm_ratio * total_steps)
        current_step  = self.state.global_step
        warm_coeff    = 1.0 if warm_steps == 0 else min(1.0, current_step / warm_steps)
        token_kl      = token_kl * warm_coeff        
        # =============================
        
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            token_kl = token_kl,                 
            # beta_eff=beta_eff,                          
        )


        kl_val = token_kl.mean().item()
        metrics["eval_KL"] = kl_val

        
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"policy_{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"policy_{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"referece_{prefix}logps/rejected"] = reference_rejected_logps.detach().cpu().mean()
        metrics[f"referece_{prefix}logps/chosen"] = reference_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits

        return losses.mean(), metrics
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
            
        loss, metrics = self.get_batch_metrics(inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss
