import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover - optional dependency
    LLM = None
    SamplingParams = None


QWEN_STOP_TOKEN_IDS = [151645, 151643]
QWEN_STOP_STRINGS = ["\nSystem:", "\nUser:", "\nAssistant:", "\nHuman:"]


class HFGenerator:
    def __init__(self, model_path: str, use_bf16: bool = False, max_model_len: int = 8192):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_model_len = max_model_len
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

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        num_return_sequences: int = 1,
        top_p: float = 1.0,
    ) -> List[str]:
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
                generation_kwargs.update({'do_sample': True, 'temperature': temperature, 'top_p': top_p})
            else:
                generation_kwargs.update({'do_sample': False})
            generated = self.model.generate(**inputs, **generation_kwargs)
            input_len = inputs.input_ids.shape[1]
            for sequence in generated:
                completion = self.tokenizer.decode(sequence[input_len:], skip_special_tokens=True).strip()
                outputs.append(completion)
        return outputs


class VLLMGenerator:
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9, max_model_len: int = 8192):
        from transformers import AutoTokenizer

        if LLM is None or SamplingParams is None:
            raise ImportError('vllm is not installed in the current environment.')
        self.max_model_len = max_model_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.llm = LLM(
            model=model_path,
            dtype='bfloat16',
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            disable_custom_all_reduce=True,
            max_model_len=max_model_len,
            enforce_eager=True,
        )

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        num_return_sequences: int = 1,
        top_p: float = 1.0,
    ) -> List[str]:
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
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


class OpenAICompletionsGenerator:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        api_base_url: str,
        api_key: str | None = None,
        max_model_len: int = 8192,
        max_concurrency: int = 8,
        timeout: float = 120.0,
        max_retries: int = 5,
    ):
        from transformers import AutoTokenizer

        if not tokenizer_path:
            raise ValueError('tokenizer_path is required for the OpenAI-compatible API backend.')
        if not api_base_url:
            raise ValueError('api_base_url is required for the OpenAI-compatible API backend.')
        if max_concurrency < 1:
            raise ValueError('max_concurrency must be >= 1')
        if max_retries < 0:
            raise ValueError('max_retries must be >= 0')

        self.model_name = model_path
        self.max_model_len = max_model_len
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key
        self.endpoint = self._completion_endpoint(api_base_url)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def _completion_endpoint(api_base_url: str) -> str:
        base_url = api_base_url.rstrip('/')
        if base_url.endswith('/completions'):
            return base_url
        if base_url.endswith('/v1'):
            return f'{base_url}/completions'
        return f'{base_url}/v1/completions'

    def _request_completion(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        num_return_sequences: int,
        top_p: float,
    ) -> List[str]:
        payload = json.dumps(
            {
                'model': self.model_name,
                'prompt': prompt,
                'max_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'n': num_return_sequences,
                'stream': False,
            }
        ).encode('utf-8')
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        for attempt in range(self.max_retries + 1):
            request = Request(self.endpoint, data=payload, headers=headers, method='POST')
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode('utf-8'))
                choices = sorted(result.get('choices', []), key=lambda choice: int(choice.get('index', 0)))
                if len(choices) != num_return_sequences:
                    raise RuntimeError(
                        f'API returned {len(choices)} choices; expected {num_return_sequences}.'
                    )
                outputs = []
                for choice in choices:
                    if 'text' not in choice:
                        raise RuntimeError('The completions API response is missing choices[].text.')
                    outputs.append(str(choice['text']).strip())
                return outputs
            except HTTPError as exc:
                error_body = exc.read().decode('utf-8', errors='replace')[:1000]
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= self.max_retries:
                    raise RuntimeError(
                        f'Completions API request failed with HTTP {exc.code}: {error_body}'
                    ) from exc
            except (URLError, TimeoutError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f'Completions API request failed after retries: {exc}') from exc
            time.sleep(min(2 ** attempt, 30))

        raise RuntimeError('Completions API request exhausted retries.')

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        num_return_sequences: int = 1,
        top_p: float = 1.0,
    ) -> List[str]:
        if not prompts:
            return []

        def generate_one(prompt: str) -> List[str]:
            return self._request_completion(
                prompt,
                max_new_tokens,
                temperature,
                num_return_sequences,
                top_p,
            )

        worker_count = min(self.max_concurrency, len(prompts))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            grouped_outputs = list(executor.map(generate_one, prompts))
        return [output for group in grouped_outputs for output in group]


def load_generator(
    backend: str,
    model_path: str,
    use_bf16: bool = False,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    tokenizer_path: str | None = None,
    api_base_url: str | None = None,
    api_key: str | None = None,
    api_max_concurrency: int = 8,
    api_timeout: float = 120.0,
    api_max_retries: int = 5,
):
    backend = backend.lower()
    if backend == 'hf':
        return HFGenerator(model_path=model_path, use_bf16=use_bf16, max_model_len=max_model_len)
    if backend == 'vllm':
        return VLLMGenerator(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
    if backend in {'api', 'openai'}:
        return OpenAICompletionsGenerator(
            model_path=model_path,
            tokenizer_path=tokenizer_path or model_path,
            api_base_url=api_base_url or '',
            api_key=api_key,
            max_model_len=max_model_len,
            max_concurrency=api_max_concurrency,
            timeout=api_timeout,
            max_retries=api_max_retries,
        )
    raise ValueError(f'Unsupported backend: {backend}')
