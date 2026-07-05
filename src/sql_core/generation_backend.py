from typing import List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover - optional dependency
    LLM = None
    SamplingParams = None


QWEN_STOP_TOKEN_IDS = [151645, 151643]
QWEN_STOP_STRINGS = ["\nSystem:", "\nUser:", "\nAssistant:", "\nHuman:"]


class HFGenerator:
    def __init__(self, model_path: str, use_bf16: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        torch_dtype = 'auto'
        if use_bf16:
            import torch
            torch_dtype = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map='auto',
        )
        self.model.eval()

    def generate_batch(self, prompts: List[str], max_new_tokens: int, temperature: float, num_return_sequences: int = 1) -> List[str]:
        outputs = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
            generation_kwargs = {
                'max_new_tokens': max_new_tokens,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'num_return_sequences': num_return_sequences,
            }
            if temperature > 0:
                generation_kwargs.update({'do_sample': True, 'temperature': temperature})
            else:
                generation_kwargs.update({'do_sample': False})
            generated = self.model.generate(**inputs, **generation_kwargs)
            input_len = inputs.input_ids.shape[1]
            for sequence in generated:
                completion = self.tokenizer.decode(sequence[input_len:], skip_special_tokens=True).strip()
                outputs.append(completion)
        return outputs


class VLLMGenerator:
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
        if LLM is None or SamplingParams is None:
            raise ImportError('vllm is not installed in the current environment.')
        self.llm = LLM(
            model=model_path,
            dtype='bfloat16',
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            disable_custom_all_reduce=True,
            max_model_len=8192,
            enforce_eager=True,
        )

    def generate_batch(self, prompts: List[str], max_new_tokens: int, temperature: float, num_return_sequences: int = 1) -> List[str]:
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=num_return_sequences,
            stop=QWEN_STOP_STRINGS,
            stop_token_ids=QWEN_STOP_TOKEN_IDS,
        )
        outputs = self.llm.generate(prompts, params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        flattened = []
        for item in outputs:
            flattened.extend(output.text for output in item.outputs)
        return flattened


def load_generator(
    backend: str,
    model_path: str,
    use_bf16: bool = False,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
):
    backend = backend.lower()
    if backend == 'hf':
        return HFGenerator(model_path=model_path, use_bf16=use_bf16)
    if backend == 'vllm':
        return VLLMGenerator(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    raise ValueError(f'Unsupported backend: {backend}')
