from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput
import torch
import torch.nn as nn

class SelfiesTedDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("ibm-research/materials.selfies-ted")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("ibm-research/materials.selfies-ted")
        self.d_model = self.model.config.d_model
        self.decoder = self.model.model.decoder

    def forward(self, custom_memory):
        encoder_outputs = BaseModelOutput(last_hidden_state=custom_memory)
        gen_ids = self.model.generate(
            encoder_outputs=encoder_outputs,
            decoder_start_token_id=self.tokenizer.bos_token_id,
            max_length=64,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        selfies_out = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return selfies_out
    