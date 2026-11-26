"""Fixed V7 reuse, wrong-model rejection and loader failure behavior."""
import io
import json
import unittest
from unittest.mock import patch

from agent.local_runtime import start_model


class LocalModelStartupChecks(unittest.TestCase):
    def test_rejects_other_adapter_in_existing_service(self):
        with patch.object(start_model.OPENER,"open",return_value=io.BytesIO(
                json.dumps({"adapter_sha256":"v6","tool_calls":True}).encode())):
            with self.assertRaisesRegex(RuntimeError,"固定 V7"):
                start_model.model_info()

    def test_ready_v7_is_reused_without_another_gpu_process(self):
        with patch.object(start_model,"model_info",return_value={"tool_calls":True}), \
                patch.object(start_model.subprocess,"Popen") as popen:
            self.assertIsNone(start_model.ensure_model())
            popen.assert_not_called()

    def test_text_only_endpoint_is_not_accepted(self):
        fingerprint=start_model.load_config()["adapter_sha256"]
        with patch.object(start_model.OPENER,"open",return_value=io.BytesIO(
                json.dumps({"adapter_sha256":fingerprint}).encode())):
            with self.assertRaisesRegex(RuntimeError,"工具协议"):
                start_model.model_info()


if __name__ == "__main__":
    unittest.main()
