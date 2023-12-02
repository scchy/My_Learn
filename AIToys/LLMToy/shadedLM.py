# python3
# Create date: 2023-11-11
# Author: Scc_hy
# Func: ShardedChatglm26B
# reference: https://www.kaggle.com/code/simjeg/platypus2-70b-without-wikipedia-rag
# ============================================================
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors.torch import load_file
import gc
import ctypes
from pathlib import Path
from optimum.bettertransformer import BetterTransformer
from threading import Condition
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
N_BATCHES = 3
MAX_LENGTH = 4096


class WeightsLoader:
    def __init__(self, checkpoint_path, devices):
        self.checkpoint_path = Path(checkpoint_path)
        self.states = {device: None for device in devices}
        self.state_dict = None
        self.condition = Condition()

    def get_state_dict(self, device):
        with self.condition:
            while self.states[device] is not None:
                self.condition.wait()

            result = self.state_dict
            self.states[device] = None 

            if not any(self.states.values()):
                self.condition.notify_all()
        
        return result
    
    def set_state_dict(self, layer_name, device):
        with self.condition:
            self.states[device] = layer_name
            if all(self.states.values()):
                assert len(set(self.states.values())) == 1, "All devices should load the same layer"
                self.state_dict = load_file(self.checkpoint_path / (layer_name + ".safetensors"), device="cpu")
                for d in self.states:
                    self.states[d] = None
                self.condition.notify_all()


def clean_memory():
    gc.collect()
    ctypes.CDLL('libc.so.6').malloc_trim(0)
    torch.cuda.empty_cache()


class ShardedChatglm26B:
    def __init__(
        self, 
        chechpoint_path, 
        weights_loader,
        device='cuda:0',
        dtype=torch.float16
    ):
        # Save paremeters
        self.chechpoint_path = Path(chechpoint_path)
        self.weights_loader = weights_loader
        self.device = device
        self.dtype = dtype

        # Create model
        self.config = AutoConfig.from_pretrained(self.chechpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.chechpoint_path, 
            trust_remote_code=True
        )
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = "right"
        self.init_model()
        self.layer_names = ["transformer.embedding"] \
                + [f"transformer.encoder.layers.{i}" 
                   for i in range(len(self.model.transformer.encoder.layers))] \
                + ["transformer.encoder.final_layernorm", "transformer.output_layer"]

    def init_model(self):
        # load meta moedl (no memroy used)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config)
            self.model.eval()
            # self.model = BetterTransformer.transform(self.model) # enable flash attention
            self.model.tie_weights()
         
        # Move buffers to device
        self.layers = [self.model.transformer.embedding] \
            + list(self.model.transformer.encoder.layers) \
            + [self.model.transformer.encoder.final_layernorm, self.model.transformer.output_layer]

        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(
                self.model, 
                buffer_name,
                self.device,
                value=buffer,
                dtype=self.dtype
            )
    
    def load_layer_to_cpu(self, layer_name):
        self.weights_loader.set_state_dict(layer_name, self.device)
        state_dict = self.weights_loader.get_state_dict(self.device)
        return state_dict

    def move_layer_to_device(self, state_dict):
        for param_name, param in state_dict.items():
            assert param.dtype != torch.int8, "int8 not supported (need to add fp16_statistics)"
            set_module_tensor_to_device(
                self.model, 
                param_name, 
                self.device, 
                value=param, 
                dtype=self.dtype
            )
    
    def __call__(self, inputs):
        del self.model
        clean_memory()
        self.init_model()

        # send batch to device
        batch = [ipts.to(self.device) for ipts in inputs]
        # Create attention mask for the largest input, and position ids to use KV cache
        attention_mask = torch.ones(MAX_LENGTH, MAX_LENGTH)
        attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0
        attention_mask = attention_mask.to(self.device)

        with ThreadPoolExecutor() as executor, torch.inference_mode():
            # Load first layer
            future = executor.submit(self.load_layer_to_cpu, "transformer.embedding")
            for i, (layer_name, layer) in tqdm(enumerate(zip(self.layer_names, self.layers)), desc=self.device, total=len(self.layers)):
                # Load current layer and prepare next layer
                state_dict = future.result()
                if (i + 1) < len(self.layer_names):
                    future = executor.submit(self.load_layer_to_cpu, self.layer_names[i + 1])
                self.move_layer_to_device(state_dict)
                
                # Run layer
                for j, ipts in enumerate(batch):
                    if layer_name == "transformer.embedding":
                        batch[j] = layer(ipts)
                    elif layer_name == "transformer.encoder.final_layernorm":
                        # Only keep the last token at this point
                        batch[j] = layer(ipts)
                    elif layer_name == "transformer.output_layer":
                        batch[j] = layer(ipts)
                    else:
                        len_p = ipts.shape[1]
                        new_prefix, (k_cache, v_cache) = layer(ipts, use_cache=True, attention_mask=attention_mask[:, :, -len_p:, -len_p:])
                        batch[j] = new_prefix

                # Remove previous layer from memory (including buffers)
                layer.to("meta")
                clean_memory() # proposed by CPMP

        # Get scores
        return batch



def get_tokens(str_in, tokenizer): 
    # max length : MAX_LENGTH
    suffix = suffix = tokenizer(
        [f"{str_in}\n\n### Response:\n"], 
        return_tensors="pt", 
        return_attention_mask=False, 
        truncation=True, 
        max_length=MAX_LENGTH, 
        padding=True
    )["input_ids"][:, 1:]
    return suffix


def run_model(device, str_in, tokenizer, checkpoint_path, weights_loader):
    model = ShardedChatglm26B(checkpoint_path, weights_loader, device=device)
    inputs = get_tokens(str_in, tokenizer)
    outputs = model(inputs)
    return outputs