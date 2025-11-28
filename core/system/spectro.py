import pytorch_lightning as pl
import numpy as np
from .base import BaseSystem
from typing import Any
from hydra.utils import instantiate
import torch
from core.metrics import SelfiesMetrics


class Spectro(BaseSystem):
    def __init__(self, config, **kwargs):
        self.tokenizer = instantiate(config.system.model.tokenizer)
        config.system.model.nmr_enc.vocab_size = self.tokenizer.get_vocab_size()
        assert (
            config.system.model.nmr_enc.vocab_size is not None
        ), "Updating vocab size failed"

        super(Spectro, self).__init__(config)
        self.save_hyperparameters()
        self.idx_to_token = self.datamodule.get_idx_to_token
        self.jirvis = instantiate(config.system.model.jirvis)
        self.nmr_enc = instantiate(config.system.model.nmr_enc)
        self.tokenizer = instantiate(config.system.model.tokenizer)

        # Initialize metrics calculator
        self.metrics_calculator = SelfiesMetrics(self.idx_to_token)

    def pre_process(self, batch):
        if hasattr(self, "data_transform") and self.data_transform is not None:
            batch = self.data_transform.pre_process(batch)
        return batch

    def validation_step(self, batch, batch_idx, **kwargs: Any):
        selfie_gt = batch["selfies"]
        tokenized_gt = batch["tokenized_selfies"]

        jirvis_embedding, nmr_embedding = self._get_embeddings(batch)

        # Use model's generate method for inference
        start_token_id = 2  # From config
        eos_token_id = 3    # From config
        max_len = tokenized_gt.shape[1]
        
        # Generate sequences
        generated_seqs = self.model.generate(
            jirvis_embedding,
            nmr_embedding,
            bos_id=start_token_id,
            eos_id=eos_token_id,
            max_new_tokens=max_len-1,
            top_p=0.9,
            temperature=1.0
        )
        
        # Remove start token from predictions for metrics
        preds = generated_seqs[:, 1:]  # [B, seq_len]
        
        # Compute validation loss using teacher forcing
        decoder_in = tokenized_gt[:, :-1]
        tgt = tokenized_gt[:, 1:]
        logits = self.model(jirvis_embedding, nmr_embedding, decoder_in)
        _, _, V = logits.shape
        loss = self.loss_func(logits.reshape(-1, V), tgt.reshape(-1))
        
        # Compute metrics
        metrics_results = self.compute_metrics(preds, selfie_gt, phase='val')
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_seq_acc", metrics_results["sequence_accuracy"]["mean"], prog_bar=True)
        self.log("val_token_acc", metrics_results["token_accuracy"]["mean"], prog_bar=True)
        # self.log("val_tanimoto", metrics_results["tanimoto_similarity"]["mean"], prog_bar=True)
        # self.log("val_valid_mol", metrics_results["valid_mol"]["mean"], prog_bar=True)
        # self.log("val_delta_hdi", metrics_results["delta_hdi"]["mean"], prog_bar=True)
        # self.log("val_size_acc", metrics_results["size_accuracy"]["mean"], prog_bar=True)
        
        return {
            "val_loss": loss,
            "metrics": metrics_results
        }

    def forward(self, batch, **kwargs):
        inputs, _ = batch
        inputs = self.pre_process(inputs)
        return self.model(inputs)

    def _get_embeddings(self, batch):
        if batch.get("images"):
            print("Using Jirvis embeddings from images in batch.")
            jirvis_embedding = self.jirvis(batch["images"])
        else:
            # print("No images found in batch, using Jirvis embeddings from batch.")
            jirvis_embedding = batch["jirvis_embeddings"]

        if (
            batch.get("input_ids")
            and batch.get("attention_mask")
            and batch.get("scalar_values")
            and batch.get("scalar_mask")
        ):
            print("Using NMR inputs from batch.")
            nmr_embedding = self.nmr_enc(
                batch["input_ids"],
                batch["attention_mask"],
                batch["scalar_values"],
                batch["scalar_mask"],
            )
        else:
            # print("No NMR inputs found in batch, using NMR embeddings from batch.")
            nmr_embedding = batch["nmr_embeddings"]

        return jirvis_embedding, nmr_embedding

    def _fuse_embeddings(self, jirvis_embedding, nmr_embedding):
        # Assuming both embeddings are of shape [B, D]
        # Concatenate along the last dimension
        fused_embedding = torch.cat((jirvis_embedding, nmr_embedding), dim=-1)
        return fused_embedding

    def training_step(self, batch, batch_idx):
        selfie_gt = batch["selfies"]
        tokenized_gt = batch["tokenized_selfies"]

        jirvis_embedding, nmr_embedding = self._get_embeddings(batch)

        decoder_in = tokenized_gt[:, :-1]
        tgt    = tokenized_gt[:, 1:]  

        logits = self.model(jirvis_embedding, nmr_embedding, decoder_in)
        _, _, V = logits.shape

        loss = self.loss_func(logits.reshape(-1, V), tgt.reshape(-1))
 

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    # def test_step(self, batch, batch_idx):
    #     selfie_gt = batch["selfies"]
    #     tokenized_gt = batch["tokenized_selfies"]

    #     jirvis_embedding, nmr_embedding = self._get_embeddings(batch)

    #     # For test, we do greedy decoding (inference)
    #     batch_size = jirvis_embedding.shape[0]
    #     max_len = tokenized_gt.shape[1]
    #     device = jirvis_embedding.device
        
    #     # Start with start token
    #     start_token_id = 2  # From config
    #     eos_token_id = 3    # From config
        
    #     # Initialize decoder input with start token
    #     decoder_input = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        
    #     # Greedy decoding
    #     for step in range(max_len - 1):
    #         logits = self.model(
    #             jirvis_embedding, nmr_embedding, decoder_input, fusion=self._fuse_embeddings
    #         )
    #         # Get last token predictions
    #         next_token_logits = logits[:, -1, :]
    #         next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
    #         # Append to decoder input
    #         decoder_input = torch.cat([decoder_input, next_tokens], dim=1)
            
    #         # Check if all sequences have generated EOS
    #         if (next_tokens.squeeze() == eos_token_id).all():
    #             break
        
    #     # Remove start token from predictions for metrics
    #     preds = decoder_input[:, 1:]  # [B, seq_len]
        
    #     # Compute test loss using teacher forcing
    #     decoder_in = tokenized_gt[:, :-1]
    #     tgt = tokenized_gt[:, 1:]
    #     logits = self.model(
    #         jirvis_embedding, nmr_embedding, decoder_in, fusion=self._fuse_embeddings
    #     )
    #     _, _, V = logits.shape
    #     loss = self.loss_func(logits.reshape(-1, V), tgt.reshape(-1))
        
    #     # Compute metrics
    #     metrics_results = self.compute_metrics(preds, selfie_gt)
        
    #     # Log metrics
    #     self.log("test_loss", loss)
    #     self.log("test_seq_acc", metrics_results["sequence_accuracy"]["mean"])
    #     self.log("test_token_acc", metrics_results["token_accuracy"]["mean"])
    #     self.log("test_tanimoto", metrics_results["tanimoto_similarity"]["mean"])
    #     self.log("test_valid_mol", metrics_results["valid_mol"]["mean"])
    #     self.log("test_delta_hdi", metrics_results["delta_hdi"]["mean"])
    #     self.log("test_size_acc", metrics_results["size_accuracy"]["mean"])
        
    #     return {
    #         "test_loss": loss,
    #         "metrics": metrics_results
    #     }

    def compute_metrics(self, preds, gt, phase="None"):
        """
        Args:
            preds: torch.Tensor [batch_size, seq_len] - predicted token IDs (logits or token IDs)
            gt: List[str] - ground truth SELFIES strings

        Returns:
            dict: Dictionary containing all computed metrics
        """
        # If preds are logits, convert to token predictions
        if preds.dim() == 3:  # [B, seq_len, vocab_size]
            preds = torch.argmax(preds, dim=-1)  # [B, seq_len]

        # Compute metrics using our SelfiesMetrics class
        results = self.metrics_calculator.compute_metrics(preds, gt, phase=phase)

        return results


    
