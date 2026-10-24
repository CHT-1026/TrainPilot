import unittest
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from app.core.config import thinking_extra_body


class ThinkingOptionsTest(unittest.TestCase):
    def test_deepseek_uses_documented_thinking_object(self):
        self.assertEqual(thinking_extra_body('deepseek-v4-flash',value='false'),{'thinking':{'type':'disabled'}})
        self.assertEqual(thinking_extra_body('deepseek-v4-flash',value='true'),{'thinking':{'type':'enabled'}})

    def test_qwen_keeps_its_own_parameter(self):
        self.assertEqual(thinking_extra_body('qwen-plus',value='false'),{'enable_thinking':False})

    def test_unspecified_flag_does_not_override_provider_default(self):
        for value in ('','auto'):
            self.assertIsNone(thinking_extra_body('deepseek-v4-flash',value=value))

    def test_routed_model_overrides_global_model_for_parameter_selection(self):
        with patch('app.core.config.get_settings',return_value=SimpleNamespace(openai_model='qwen-plus',openai_enable_thinking='false')):
            self.assertEqual(thinking_extra_body('deepseek-v4-flash'),{'thinking':{'type':'disabled'}})

    def test_standalone_model_check_honors_file_proxy_and_deepseek_parameter(self):
        from local_runtime import check_model
        values={'OPENAI_API_KEY':'fake','OPENAI_API_BASE':'https://gateway.example.invalid/v1',
                'OPENAI_MODEL':'deepseek-v4-flash','OPENAI_ENABLE_THINKING':'false',
                'NO_PROXY':'gateway.example.invalid'}
        opener=MagicMock()
        opener.open.return_value=io.BytesIO(json.dumps({'choices':[{'message':{'content':'OK'}}]}).encode())
        with tempfile.TemporaryDirectory() as folder,patch.object(check_model,'REPORT',Path(folder)/'report.json'), \
            patch.object(check_model,'dotenv_values',return_value=values), \
            patch('urllib.request.build_opener',return_value=opener), \
            patch('urllib.request.urlopen',side_effect=AssertionError('Proxy path must not be used')):
            self.assertTrue(check_model.check()['ok'])
        body=json.loads(opener.open.call_args.args[0].data)
        self.assertEqual(body['thinking'],{'type':'disabled'})
        self.assertNotIn('enable_thinking',body)
