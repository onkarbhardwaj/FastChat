import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from fastchat.model.base_model_adapter import BaseModelAdapter
from fastchat.conversation import Conversation, get_conv_template
from fastchat.model.compression import load_compress_model


class IbmLabradoriteAdapter(BaseModelAdapter):
    """The base and the default model adapter."""

    # use_fast_tokenizer = True

    def match(self, model_path: str):
        print(f"IBM Labradorite adapter: {'labradorite' in model_path.lower()}")
        return "labradorite" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("labradorite")


class IbmModelBigCodeAdapter(BaseModelAdapter):
    """The base and the default model adapter."""
    print("Trying IBM BigCode adapter")

    use_fast_tokenizer = True

    def match(self, model_path: str):
        return "ibm" in model_path.lower() and "bigcode" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        print("Loading from IBM BigCode adapter")
        try:
            # Changed from torch.bloat16 to torch.float16
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
            model.eval()
        except Exception as e:
            raise ValueError(f"AutoModelForCausalLM is not found or unable to load the model: {e}")
        else:
            return model, tokenizer

    def load_compress_model(self, model_path, device, torch_dtype, revision="main"):
        return load_compress_model(
            model_path,
            device,
            torch_dtype,
            use_fast=self.use_fast_tokenizer,
            revision=revision,
        )

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("ibm-labrador")


class IbmModelMegatronAdapter(BaseModelAdapter):
    """The base and the default model adapter."""
    print("Trying IBM Megatron adapter")

    use_fast_tokenizer = True

    def match(self, model_path: str):
        return "ibm" in model_path.lower() and "megatron" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        try:
            from ibm_models import GPTMegatronForCausalLM
        except:
            raise Exception("ibm-models package not installed, please install separately.")
        
        revision = from_pretrained_kwargs.get("revision", "main")
        print("Loading from IBM Megatron adapter")
        try:
            model = GPTMegatronForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
            model.eval()
        except Exception as e:
            raise ValueError(f"GPTMegatronForCausalLM is not found or unable to load the model: {e}")
        else:
            return model, tokenizer

    def load_compress_model(self, model_path, device, torch_dtype, revision="main"):
        return load_compress_model(
            model_path,
            device,
            torch_dtype,
            use_fast=self.use_fast_tokenizer,
            revision=revision,
        )

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("ibm-labrador")


class IbmModelDolomiteAdapter(BaseModelAdapter):
    """The base and the default model adapter."""
    print("Trying IBM Dolomite adapter")

    use_fast_tokenizer = True

    def match(self, model_path: str):
        return "ibm" in model_path.lower() and "dolomite" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        try:
            from dolomite_engine.hf_models import GPTDolomiteForCausalLM
        except:
            raise Exception("dolomite_engine package not installed, please install separately.")
        
        revision = from_pretrained_kwargs.get("revision", "main")
        print("Loading from IBM Megatron adapter")
        try:
            model = GPTDolomiteForCausalLM.from_pretrained(
                model_path,
                attn_implementation="flash_attention_2",
                use_padding_free_transformer=False,
                trust_remote_code=True, # Might need to be removed
                torch_dtype=torch.bfloat16 # Might need to be removed
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
            model.eval()
        except Exception as e:
            raise ValueError(f"GPTDolomiteForCausalLM is not found or unable to load the model: {e}")
        else:
            return model, tokenizer

    def load_compress_model(self, model_path, device, torch_dtype, revision="main"):
        return load_compress_model(
            model_path,
            device,
            torch_dtype,
            use_fast=self.use_fast_tokenizer,
            revision=revision,
        )

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("ibm-labrador")