import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput
import json
import os
import torch.nn.functional as F

import sys
sys.path.append("../downstream_example")
from eva_clip import create_eva_vision_and_transforms

from torch.utils.tensorboard import SummaryWriter
from clip_loss import ClipLoss

logger = logging.getLogger(__name__)

def pairwise_distance(a, b):
    diff = a.unsqueeze(1) - b.unsqueeze(0)
    dist = (diff ** 2).sum(dim=-1).sqrt()
    return dist

def cus_cdist(A,B):
    '''
        A: batch,embsz
        B: batch,embsz
    '''
    D = A @ B.T
    D = 1.0 - D
    return D

def all_gather_tensors(tensor):

    tensor = tensor.contiguous()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_size = torch.tensor(tensor.size(0), device=tensor.device)
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, local_size)

    max_size = max(gathered_sizes).item()
    padding_size = max_size - tensor.size(0)
    if padding_size > 0:
        padding = torch.zeros(padding_size, tensor.size(1), device=tensor.device, dtype=tensor.dtype)
        tensor_padded = torch.cat((tensor, padding), dim=0)
    else:
        tensor_padded = tensor

    gathered_tensors = [torch.zeros_like(tensor_padded) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor_padded)
    gathered_tensors = [g[:size.item()] for g, size in zip(gathered_tensors, gathered_sizes)]
    result_tensor = torch.cat(gathered_tensors, dim=0)
    result_tensor.requires_grad_(True)
    return result_tensor

@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    c_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class mlp(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, 4 * input_dim)
        self.fc2 = nn.Linear(4 * input_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, output_dim)

        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

class BGE_EVAToken(nn.Module):
    '''
    BGE + CLIP-V
    '''
    
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name_bge: str = None,
                 model_name_eva: str = "EVA02-CLIP-B-16",
                 normlized: bool = True, #False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = True,
                 temperature: float = 0.02, # 1.0
                 eva_pretrained_path = None,
                 writer:SummaryWriter = None
                 ):
        super().__init__()

        if eva_pretrained_path is None:
            eva_pretrained_path = "eva_clip" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"

        self.expand = True
        self.bge_encoder = AutoModel.from_pretrained(model_name_bge).encoder
        self.bge_embeddings = AutoModel.from_pretrained(model_name_bge).embeddings
        self.bge_pooler = AutoModel.from_pretrained(model_name_bge).pooler
                
        self.model_visual, self.preprocess_train, self.preprocess_val= create_eva_vision_and_transforms(
            model_name_eva, 
            eva_pretrained_path, 
            force_custom_clip=True)

        self.visual_proj = nn.Linear(768, 768)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            print('process_rank : ', self.process_rank)
            print('world_size : ', self.world_size)
        self.tf_writer = writer

    def gradient_checkpointing_enable(self, **kwargs):
        # self.bge_encoder.gradient_checkpointing_enable()
        self.model_visual.set_grad_checkpointing(True)
    
    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = torch.float16
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        
        return extended_attention_mask

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    
    
    def t_encoder(self, texts):
        '''
        encode captions only, use for training
        '''
        input_ids = texts['input_ids']
        attention_mask = texts['attention_mask']

        input_shape = input_ids.size()
        device = input_ids.device

        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        head_mask = [None] * 12
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        
        embedding_output = self.bge_embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )
        encoder_outputs = self.bge_encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.bge_pooler(sequence_output) if self.bge_pooler is not None else None

        t_reps = self.sentence_embedding(sequence_output, texts['attention_mask']) # tensor: reps with pooling
        if self.normlized:
            t_reps = torch.nn.functional.normalize(t_reps, dim=-1)
        return t_reps.contiguous()

    def mm_encoder(self, images, prompts):
        img_token_emb = self.img_token_embedding(images) #[B, T, C]
        if img_token_emb is not None:
            img_token_emb = img_token_emb[:,1:]
            img_token_emb = self.visual_proj(img_token_emb)
            device = img_token_emb.device
            
            img_token_len = img_token_emb.size()[1]

            # image position embedding
            img_token_position_ids = torch.arange(1, 1 + img_token_len).to(device=device)
            img_position_embeddings = self.bge_embeddings.position_embeddings(img_token_position_ids)
            img_token_emb = img_token_emb + img_position_embeddings

            img_token_emb = self.bge_embeddings.LayerNorm(img_token_emb)
        else:
            return self.t_encoder(prompts)

        prompt_input_ids = prompts['input_ids']
        prompt_attention_mask = prompts['attention_mask']
        prom_input_shape = prompt_input_ids.size()
        
        batch_size = prom_input_shape[0]
        prompt_len = prom_input_shape[1]
        prompt_start = 1 + img_token_len

        
        
        cls_id = torch.tensor([0]).to(device=device)
        prompt_position_ids = torch.arange(prompt_start, prompt_start + prompt_len - 1).to(device=device)
        prompt_position_ids = torch.cat([cls_id, prompt_position_ids]).to(device=device)

        prompt_token_type_ids = torch.zeros(prom_input_shape, dtype=torch.long, device=device)
        prompt_embedding_output = self.bge_embeddings(
            input_ids=prompt_input_ids,
            position_ids=prompt_position_ids,
            token_type_ids=prompt_token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )  # [B, T, C]

        ### prompt+img embeddings --> encoder
        cls_token = prompt_embedding_output[:, 0:1, :]
        prompt_embedding_output = prompt_embedding_output[:, 1:]

        prompt_img_embedding = torch.cat([cls_token, img_token_emb, prompt_embedding_output], dim=1) 
        
        img_attention_mask = torch.ones(batch_size, img_token_len, device=device)  
        prom_img_attention_mask = torch.cat([img_attention_mask, prompt_attention_mask], dim=1)
        prom_img_input_shape = prompt_img_embedding.size()

        head_mask = [None] * 12
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(prom_img_attention_mask, prom_img_input_shape)
        
        encoder_outputs = self.bge_encoder(
            prompt_img_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]
        
        prompt_img_reps = self.sentence_embedding(sequence_output, prom_img_attention_mask) # tensor: reps with pooling
        if self.normlized:
            prompt_img_reps = torch.nn.functional.normalize(prompt_img_reps, dim=-1)
        return prompt_img_reps

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def img_token_embedding(self, images):
        '''
        Used for training
        '''
        if images is None:
            return None
        img_token_emb = self.model_visual.encode_image(images, normalize=False) #return_all_features=True, [B, T, 768] 
        
        return img_token_emb.contiguous()
    
    def encode_text(self, texts):
        '''
        now, this function is used for CLIP_Benchmarking
        return pooling + linear adaptor features of texts but without normalization
        '''
        
        if texts is None:
            return None
        
        return self.t_encoder(texts)
    
    def encode_image_only(self, images, prompts = None):
        '''
        now, this function is used for Retrival Benchmarking
        the difference with encode_image is no prompts, just a bge cls
        '''
        if images is None:
            return None
        
        if prompts is None:
            prompts = {'input_ids': torch.tensor([101, 102]),  
          'attention_mask': torch.tensor([1,1])}
            device = images.device
            batch_size = images.shape[0]
            prompt_input_ids = prompts['input_ids'].to(device).unsqueeze(0).repeat(batch_size, 1)
            prompt_attention_mask = prompts['attention_mask'].to(device).unsqueeze(0).repeat(batch_size, 1)
            prompts = {'input_ids': prompt_input_ids,  
                       'attention_mask': prompt_attention_mask}
        
        img_reps = self.mm_encoder(images, prompts)
        return img_reps
    
    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def forward(self, item_images=None, item_texts=None, task_type="i2it",hard_neg_num=None,loss_fn=None, query_images=None, query_texts=None,neg_item_texts=None,neg_item_images=None,
                origin_query=None,origin_prod=None,origin_text=None,labels=None,margin=None, is_suploss=False):

        if task_type == "clip_contrasive":
            item_images = self.encode_image_only(item_images)
            item_texts = self.encode_text(item_texts)
            
            if self.negatives_cross_device:
                item_images = self._dist_gather_tensor(item_images)
                item_texts = self._dist_gather_tensor(item_texts)
            
            cosines = item_texts @ item_images.T / self.temperature
           
            target = torch.arange(item_images.size(0), device=item_images.device, dtype=torch.long)
            target = target * (item_images.size(0) // item_images.size(0))
            
            clip_loss = (self.compute_loss(cosines, target) + self.compute_loss(cosines.T, target)) / 2
            loss = clip_loss
            self.tf_writer.add_scalar("clip_loss", clip_loss.cpu().item())
            logging.info("task types: %s; loss: %s" %(task_type, str(clip_loss.cpu().item())))
            
            return EncoderOutput(
                loss=loss,
                scores=cosines,
                q_reps=item_images,
                c_reps=item_texts,
            )
                    elif task_type == "t2it":
            item_multimodel = self.mm_encoder(item_images, item_texts)
            query_texts = self.encode_text(query_texts)
            if self.negatives_cross_device:
                item_multimodel = self._dist_gather_tensor(item_multimodel)
                query_texts = self._dist_gather_tensor(query_texts)
            
            cosines = self.compute_similarity(query_texts, item_multimodel)
            cosines = cosines / self.temperature
            cosines = cosines.view(query_texts.size(0), -1)
            
            target = torch.arange(cosines.size(0), device=cosines.device, dtype=torch.long)
            # target = target * (candi_reps.size(0) // query_reps.size(0))
            
            clip_loss = self.compute_loss(cosines, target) + self.compute_loss(cosines.T, target)
            loss = clip_loss
            self.tf_writer.add_scalar("t2it_loss", clip_loss)
            logging.info("task types: %s; loss: %s" %(task_type, str(clip_loss)))
            
            return EncoderOutput(
                loss=loss,
                scores=cosines,
                q_reps=item_images,
                c_reps=item_texts,
            )
            
        elif task_type == "i2it": 
            item_multimodel = self.mm_encoder(item_images, item_texts)
            query_images = self.encode_image_only(query_images)
            if self.negatives_cross_device:
                item_multimodel = self._dist_gather_tensor(item_multimodel)
                query_images = self._dist_gather_tensor(query_images)
            
            cosines = query_images @ item_multimodel.T / self.temperature
            cosines = cosines.view(query_images.size(0), -1)
            
            target = torch.arange(cosines.size(0), device=cosines.device, dtype=torch.long)
            clip_loss = (self.compute_loss(cosines, target) + self.compute_loss(cosines.T, target)) / 2
            sup_loss = 0.0
            # import pdb
            # pdb.set_trace()
            if is_suploss:
                query_dis = cus_cdist(query_images,query_images) # (batch_size, batch_size)
                mm_dis = cus_cdist(item_multimodel,item_multimodel) # (batch_size, batch_size)
                sup_losses = (torch.relu(1.0 - query_dis) + torch.relu(1.0 - mm_dis)) / 2.0
                sup_loss = sup_losses.mean()
                loss = losses.mean() * 0.8 + 0.2 * sup_loss
            else:
                # loss = losses.mean()
                loss = clip_loss
            self.tf_writer.add_scalar("clip_loss", clip_loss.cpu().item())
            logging.info("task types: %s; loss: %s" %(task_type, str(clip_loss.cpu().item())))
            return EncoderOutput(
                loss=loss,
                scores=cosines,
                q_reps=item_images,
                c_reps=item_texts,
            )
        elif task_type == "stage3":
            pos_item_multimodel = self.mm_encoder(item_images, item_texts)
            query_images = self.encode_image_only(query_images)
            if self.negatives_cross_device:
                pos_item_multimodel = self._dist_gather_tensor(pos_item_multimodel) # # b, emb_size
                query_images = self._dist_gather_tensor(query_images) # b, emb_size

            n = item_images.size(0)
            if loss_fn == 'triplet_cos':
        
                neg_item_multimodel = self.mm_encoder(neg_item_images, neg_item_texts)  # （batch,emb_sz)
                if self.negatives_cross_device:
                    neg_item_multimodel = self._dist_gather_tensor(neg_item_multimodel)

                targets = torch.zeros(query_images.size(0), dtype=torch.long, device=query_images.device)

                candi_reps = torch.stack([pos_item_multimodel,neg_item_multimodel],dim=1)
                scores = self.compute_similarity(query_images, candi_reps)
                scores = scores / self.temperature
                scores = scores.view(query_images.size(0), -1)
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * 2
                
                loss_edit = self.compute_loss(scores, target)
                loss = loss_edit


            elif loss_fn == 'triplet_dist':
                
                neg_item_multimodel = self.mm_encoder(neg_item_images, neg_item_texts)  # （batch,emb_sz)
                if self.negatives_cross_device:
                    neg_item_multimodel = self._dist_gather_tensor(neg_item_multimodel)
                # positive_distances = F.pairwise_distance(query_images.float(), pos_item_multimodel.float()).squeeze() # (batch_size,)
                positive_distances = (((query_images - pos_item_multimodel) ** 2).sum(dim=1) + 1e-5).sqrt()

            
                # negative_distances = F.pairwise_distance(query_images.float(), neg_item_multimodel.float().squeeze()).squeeze()  # (batch_size, num_negatives)
                negative_distances = (((query_images - neg_item_multimodel) ** 2).sum(dim=1) + 1e-5).sqrt()
                
                losses = torch.relu(positive_distances - negative_distances + margin)  # (batch_size, )

                sup_loss = 0.0
                # import pdb
                # pdb.set_trace()
                if is_suploss:
                    query_dis = cus_cdist(query_images,query_images) # (batch_size, batch_size)
                    mm_pos_dis = cus_cdist(pos_item_multimodel,pos_item_multimodel) # (batch_size, batch_size)
                    mm_neg_dis = cus_cdist(neg_item_multimodel,neg_item_multimodel) # (batch_size, batch_size)
                    sup_losses = (torch.relu(1.0 - query_dis) + torch.relu(1.0 - mm_pos_dis) + torch.relu(1.0 - mm_neg_dis)) / 3.0
                    sup_loss = sup_losses.mean()
                    loss = losses.mean() * 0.8 + 0.2 * sup_loss
                else:
                    loss = losses.mean()
                # cosines = positive_distances
                
            elif loss_fn == 'cross_loss':
                
                prod_item_multimodel = pos_item_multimodel
                similarities = F.cosine_similarity(query_images, prod_item_multimodel, dim=-1).squeeze().float()  # (batch_size,)
                labels = torch.tensor(labels, dtype=torch.float32, device=similarities.device)
                loss = F.binary_cross_entropy(similarities,labels)

            self.tf_writer.add_scalar("stage3_loss", loss.cpu().item())
            logging.info("task types: %s; loss fn : %s; loss: %s" %(task_type,loss_fn, str(loss.cpu().item())))
           
            return EncoderOutput(
                loss=loss,
            )
        else:
            logging.info("please give a valid task id")
            return None 
    

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        torch.save(self.state_dict(), os.path.join(output_dir, 'BGE_EVA_Token.pth'))