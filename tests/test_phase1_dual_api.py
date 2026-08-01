import json
import sys
import types
import unittest
from unittest.mock import patch

from scripts.srt.generate_phase1_traces import (
    build_generator_config,
    generate_revised_candidates,
    init_summary_state,
    materialize_summary,
    parse_args,
    resolve_generation_args,
)
from src.phase1_srt.constants import P_R_INCORRECT
from src.sql_core.generation_backend import OpenAICompletionsGenerator


class FakeTokenizer:
    eos_token = '<eos>'
    eos_token_id = 1
    pad_token = None
    pad_token_id = None

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': list(text)}


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.body).encode('utf-8')


class RecordingGenerator:
    max_model_len = 8192

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.prompts = []

    def generate_batch(
        self,
        prompts,
        max_new_tokens,
        temperature,
        num_return_sequences=1,
        top_p=1.0,
    ):
        self.prompts.extend(prompts)
        return [
            f'SELECT {index}'
            for _prompt in prompts
            for index in range(num_return_sequences)
        ]


class OpenAICompletionsGeneratorTests(unittest.TestCase):
    def test_raw_completions_request_preserves_choice_order(self):
        fake_transformers = types.ModuleType('transformers')
        fake_transformers.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: FakeTokenizer()
        )
        response = FakeResponse(
            {
                'choices': [
                    {'index': 1, 'text': ' SELECT 2 '},
                    {'index': 0, 'text': ' SELECT 1 '},
                ]
            }
        )
        with patch.dict(sys.modules, {'transformers': fake_transformers}):
            generator = OpenAICompletionsGenerator(
                model_path='api-model',
                tokenizer_path='local-tokenizer',
                api_base_url='https://example.test/v1',
                api_key='secret',
                max_concurrency=1,
            )
        with patch('src.sql_core.generation_backend.urlopen', return_value=response) as mocked_urlopen:
            outputs = generator.generate_batch(
                ['rendered prompt'],
                max_new_tokens=64,
                temperature=0.7,
                num_return_sequences=2,
                top_p=0.9,
            )

        self.assertEqual(outputs, ['SELECT 1', 'SELECT 2'])
        request = mocked_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode('utf-8'))
        self.assertEqual(request.full_url, 'https://example.test/v1/completions')
        self.assertEqual(request.get_header('Authorization'), 'Bearer secret')
        self.assertEqual(payload['prompt'], 'rendered prompt')
        self.assertEqual(payload['model'], 'api-model')
        self.assertEqual(payload['n'], 2)


class DualGeneratorConfigurationTests(unittest.TestCase):
    def test_old_summary_state_uses_legacy_backend_fallback(self):
        with patch.object(sys, 'argv', ['generate_phase1_traces.py']), patch.dict('os.environ', {}, clear=True):
            args = resolve_generation_args(parse_args())
        state = init_summary_state(args, selected_count=1, shard_count=1)
        state['backend'] = 'vllm'
        for field in ('init_backend', 'revision_backend', 'init_model', 'revision_model'):
            state.pop(field)

        summary = materialize_summary(state)

        self.assertEqual(summary['init_backend'], 'vllm')
        self.assertEqual(summary['revision_backend'], 'vllm')
        self.assertIsNone(summary['init_model'])
        self.assertIsNone(summary['revision_model'])

    def test_phase_specific_api_models_and_keys_are_resolved(self):
        argv = [
            'generate_phase1_traces.py',
            '--init-backend',
            'api',
            '--revision-backend',
            'api',
            '--init-model-path',
            'Qwen3-4B-Instruct-2507',
            '--revision-model-path',
            'Qwen3-Coder-30B-A3B-Instruct',
            '--init-tokenizer-path',
            '/models/qwen3-tokenizer',
            '--api-base-url',
            'https://example.test/v1',
        ]
        environment = {
            'PHASE1_INIT_API_KEY': 'init-secret',
            'PHASE1_REVISION_API_KEY': 'revision-secret',
        }
        with patch.object(sys, 'argv', argv), patch.dict('os.environ', environment, clear=True):
            args = resolve_generation_args(parse_args())
            init_config = build_generator_config(args, 'init')
            revision_config = build_generator_config(args, 'revision')

        self.assertEqual(init_config['model_path'], 'Qwen3-4B-Instruct-2507')
        self.assertEqual(revision_config['model_path'], 'Qwen3-Coder-30B-A3B-Instruct')
        self.assertEqual(init_config['api_key'], 'init-secret')
        self.assertEqual(revision_config['api_key'], 'revision-secret')
        self.assertEqual(init_config['api_base_url'], 'https://example.test/v1')
        self.assertEqual(revision_config['api_base_url'], 'https://example.test/v1')

    def test_revision_continues_the_init_models_rendered_prompt(self):
        generator = RecordingGenerator()
        stage_rows = [
            {
                'sample': {'id': 'q1', 'db_id': 'db'},
                'base_prompt': '<qwen4b-chat>assistant\n',
                'init_index': 0,
                'y_init': 'SELECT missing FROM table_name',
                'verifier_init': {'reward': 0},
            }
        ]
        state = {
            'prompt_max_observed_token_length': 0,
            'prompt_overflow_count': 0,
            'prompt_overflow_sample_count': 0,
            'prompt_overflow_init_count': 0,
            'prompt_overflow_revision_count': 0,
            'prompt_overflow_examples': [],
        }

        generate_revised_candidates(
            stage_rows,
            generator=generator,
            batch_size=1,
            max_new_tokens=256,
            temperature=0.7,
            top_p=1.0,
            num_revisions=3,
            state=state,
        )

        expected_prompt = (
            '<qwen4b-chat>assistant\nSELECT missing FROM table_name\n\n'
            f'{P_R_INCORRECT}\n\n'
        )
        self.assertEqual(generator.prompts, [expected_prompt])
        self.assertEqual([row['revision_index'] for row in stage_rows], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
