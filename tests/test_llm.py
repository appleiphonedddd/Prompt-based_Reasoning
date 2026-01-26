import unittest
from unittest.mock import MagicMock, patch
import os

from models.base import LLMResponse, BaseLLM
from models.deepseek import DeepSeekClient
from models.gpt import GPTClient
from models.gemini import GeminiClient
from models.llama import LlamaClient


class TestLLMClasses(unittest.TestCase):

    def setUp(self):
        self.fake_api_key = "sk-fake-key-123"

    # ---------- Base / Interface tests ----------

    def test_base_llm_is_abstract(self):
        # BaseLLM has an abstract method generate -> cannot instantiate
        with self.assertRaises(TypeError):
            BaseLLM(api_key="x", model="y")

    # ---------- DeepSeek tests ----------

    @patch('models.deepseek.OpenAI')
    def test_deepseek_initialization(self, MockOpenAI):
        client = DeepSeekClient(api_key=self.fake_api_key)
        MockOpenAI.assert_called_with(api_key=self.fake_api_key, base_url="https://api.deepseek.com")
        self.assertEqual(client.model, "deepseek-chat")

    @patch('models.deepseek.OpenAI')
    def test_deepseek_generate_success(self, MockOpenAI):
        mock_instance = MockOpenAI.return_value

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "DeepSeek response content"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.model_dump.return_value = {"id": "123"}

        mock_instance.chat.completions.create.return_value = mock_response

        client = DeepSeekClient(api_key=self.fake_api_key)
        result = client.generate("Hello")

        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.content, "DeepSeek response content")
        self.assertEqual(result.input_tokens, 10)
        self.assertEqual(result.output_tokens, 20)
        self.assertEqual(result.model_name, "deepseek-chat")

    @patch('models.deepseek.OpenAI')
    def test_deepseek_temperature_forwarding(self, MockOpenAI):
        mock_instance = MockOpenAI.return_value
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1
        mock_response.model_dump.return_value = {}

        mock_instance.chat.completions.create.return_value = mock_response

        client = DeepSeekClient(api_key=self.fake_api_key)
        client.generate("Hello", temperature=0.7)

        mock_instance.chat.completions.create.assert_called_once()
        _, kwargs = mock_instance.chat.completions.create.call_args
        self.assertEqual(kwargs["temperature"], 0.7)

    @patch('models.deepseek.OpenAI')
    def test_deepseek_api_runtime_error(self, MockOpenAI):
        mock_instance = MockOpenAI.return_value
        mock_instance.chat.completions.create.side_effect = Exception("Network connection error")

        client = DeepSeekClient(api_key=self.fake_api_key)
        with self.assertRaises(RuntimeError) as cm:
            client.generate("Test error")

        self.assertIn("DeepSeek API Error", str(cm.exception))

    def test_deepseek_missing_api_key_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                DeepSeekClient(api_key=None)

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-deepseek-key"}, clear=True)
    @patch('models.deepseek.OpenAI')
    def test_deepseek_env_api_key_fallback(self, MockOpenAI):
        client = DeepSeekClient(api_key=None)
        MockOpenAI.assert_called_with(api_key="env-deepseek-key", base_url="https://api.deepseek.com")
        self.assertEqual(client.api_key, "env-deepseek-key")

    # ---------- Llama tests ----------

    @patch('models.llama.OpenAI')
    def test_llama_initialization(self, MockOpenAI):
        client = LlamaClient(api_key=self.fake_api_key)
        MockOpenAI.assert_called_with(api_key=self.fake_api_key, base_url="https://api.llama.com/compat/v1/")
        self.assertEqual(client.model, "Llama-3.3-8B-Instruct")

    @patch('models.llama.OpenAI')
    def test_llama_generate_success(self, MockOpenAI):
        mock_instance = MockOpenAI.return_value

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Llama response content"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.model_dump.return_value = {"id": "123"}

        mock_instance.chat.completions.create.return_value = mock_response

        client = LlamaClient(api_key=self.fake_api_key)
        result = client.generate("Hello")

        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.content, "Llama response content")
        self.assertEqual(result.input_tokens, 10)
        self.assertEqual(result.output_tokens, 20)
        self.assertEqual(result.model_name, "Llama-3.3-8B-Instruct")

    @patch('models.llama.OpenAI')
    def test_llama_temperature_forwarding(self, MockOpenAI):
        mock_instance = MockOpenAI.return_value
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1
        mock_response.model_dump.return_value = {}

        mock_instance.chat.completions.create.return_value = mock_response

        client = LlamaClient(api_key=self.fake_api_key)
        client.generate("Hello", temperature=0.3)

        mock_instance.chat.completions.create.assert_called_once()
        _, kwargs = mock_instance.chat.completions.create.call_args
        self.assertEqual(kwargs["temperature"], 0.3)

    @patch('models.llama.OpenAI')
    def test_llama_api_runtime_error(self, MockOpenAI):
        mock_instance = MockOpenAI.return_value
        mock_instance.chat.completions.create.side_effect = Exception("Network connection error")

        client = LlamaClient(api_key=self.fake_api_key)
        with self.assertRaises(RuntimeError) as cm:
            client.generate("Test error")

        self.assertIn("Llama API Error", str(cm.exception))

    def test_llama_missing_api_key_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                LlamaClient(api_key=None)

    @patch.dict(os.environ, {"LLAMA_API_KEY": "env-llama-key"}, clear=True)
    @patch('models.llama.OpenAI')
    def test_llama_env_api_key_fallback(self, MockOpenAI):
        client = LlamaClient(api_key=None)
        MockOpenAI.assert_called_with(api_key="env-llama-key", base_url="https://api.llama.com/compat/v1/")
        self.assertEqual(client.api_key, "env-llama-key")

    # ---------- GPT tests ----------

    @patch('models.gpt.OpenAI')
    def test_gpt_initialization(self, MockOpenAI):
        client = GPTClient(api_key=self.fake_api_key)
        MockOpenAI.assert_called_with(api_key=self.fake_api_key)
        self.assertEqual(client.model, "gpt-4o-mini")

    @patch('models.gpt.OpenAI')
    def test_gpt_generate_success(self, MockOpenAI):
        mock_instance = MockOpenAI.return_value

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "GPT response content"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 5
        mock_response.model_dump.return_value = {}

        mock_instance.chat.completions.create.return_value = mock_response

        client = GPTClient(api_key=self.fake_api_key)
        result = client.generate("Hello GPT")

        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.content, "GPT response content")
        self.assertEqual(result.model_name, "gpt-4o-mini")

    @patch('models.gpt.OpenAI')
    def test_gpt_temperature_forwarding(self, MockOpenAI):
        mock_instance = MockOpenAI.return_value
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1
        mock_response.model_dump.return_value = {}

        mock_instance.chat.completions.create.return_value = mock_response

        client = GPTClient(api_key=self.fake_api_key)
        client.generate("Hello", temperature=0.9)

        mock_instance.chat.completions.create.assert_called_once()
        _, kwargs = mock_instance.chat.completions.create.call_args
        self.assertEqual(kwargs["temperature"], 0.9)

    def test_gpt_missing_api_key_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                GPTClient(api_key=None)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"}, clear=True)
    @patch('models.gpt.OpenAI')
    def test_gpt_env_api_key_fallback(self, MockOpenAI):
        client = GPTClient(api_key=None)
        MockOpenAI.assert_called_with(api_key="env-openai-key")
        self.assertEqual(client.api_key, "env-openai-key")

    @patch('models.gpt.OpenAI')
    def test_gpt_api_runtime_error(self, MockOpenAI):
        mock_instance = MockOpenAI.return_value
        mock_instance.chat.completions.create.side_effect = Exception("Network connection error")

        client = GPTClient(api_key=self.fake_api_key)

        with self.assertRaises(RuntimeError) as cm:
            client.generate("Test error")

        self.assertIn("GPT API Error", str(cm.exception))

    # ---------- Gemini tests ----------

    @patch('models.gemini.genai.Client')
    def test_gemini_initialization(self, MockGenAIClient):
        client = GeminiClient(api_key=self.fake_api_key)
        MockGenAIClient.assert_called_with(api_key=self.fake_api_key)
        self.assertEqual(client.model, "gemini-2.0-flash-lite")

    @patch('models.gemini.genai.Client')
    def test_gemini_generate_success(self, MockGenAIClient):
        mock_instance = MockGenAIClient.return_value

        mock_response = MagicMock()
        mock_response.text = "Gemini response content"
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 200

        mock_instance.models.generate_content.return_value = mock_response

        client = GeminiClient(api_key=self.fake_api_key)
        result = client.generate("Hello Gemini")

        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.content, "Gemini response content")
        self.assertEqual(result.input_tokens, 100)
        self.assertEqual(result.output_tokens, 200)
        self.assertEqual(result.model_name, "gemini-2.0-flash-lite")

    @patch('models.gemini.types.GenerateContentConfig')
    @patch('models.gemini.genai.Client')
    def test_gemini_temperature_forwarding(self, MockGenAIClient, MockGenerateContentConfig):
        mock_instance = MockGenAIClient.return_value

        mock_config_obj = MagicMock()
        MockGenerateContentConfig.return_value = mock_config_obj

        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_response.usage_metadata.prompt_token_count = 1
        mock_response.usage_metadata.candidates_token_count = 1
        mock_instance.models.generate_content.return_value = mock_response

        client = GeminiClient(api_key=self.fake_api_key)
        client.generate("Hello", temperature=0.42)

        MockGenerateContentConfig.assert_called_once()
        _, kwargs = MockGenerateContentConfig.call_args
        self.assertEqual(kwargs["temperature"], 0.42)

        mock_instance.models.generate_content.assert_called_once()
        _, kwargs2 = mock_instance.models.generate_content.call_args
        self.assertEqual(kwargs2["config"], mock_config_obj)

    def test_gemini_missing_api_key_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                GeminiClient(api_key=None)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "env-gemini-key"}, clear=True)
    @patch('models.gemini.genai.Client')
    def test_gemini_env_api_key_fallback(self, MockGenAIClient):
        client = GeminiClient(api_key=None)
        MockGenAIClient.assert_called_with(api_key="env-gemini-key")
        self.assertEqual(client.api_key, "env-gemini-key")

    @patch('models.gemini.genai.Client')
    def test_gemini_api_runtime_error(self, MockGenAIClient):
        mock_instance = MockGenAIClient.return_value
        mock_instance.models.generate_content.side_effect = Exception("Network connection error")

        client = GeminiClient(api_key=self.fake_api_key)
        with self.assertRaises(RuntimeError) as cm:
            client.generate("Test error")

        self.assertIn("Gemini API Error", str(cm.exception))


if __name__ == '__main__':
    unittest.main()
