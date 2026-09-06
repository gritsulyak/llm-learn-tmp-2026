import json
import os
import urllib.request
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

        self.max_new_tokens = int(self._get_param("max_new_tokens"))
        self.temperature = float(self._get_param("temperature"))

        self.server_url = os.environ.get(
            "LLAMA_SERVER_URL", "http://host.docker.internal:8080"
        ).rstrip("/")
        self._hf = None

    def _get_param(self, name):
        params = self.model_config.get("parameters", {})
        return params.get(name, {}).get("string_value", "")

    def _generate_llama_cpp(self, prompt):
        body = json.dumps(
            {
                "prompt": prompt,
                "n_predict": self.max_new_tokens,
                "temperature": self.temperature,
                "cache_prompt": True,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.server_url + "/completion",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["content"]

    def _generate_hf_fallback(self, prompt):
        if self._hf is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            name = self._get_param("hf_model_name") or "HuggingFaceTB/SmolLM-135M"
            self._hf = (
                AutoTokenizer.from_pretrained(name),
                AutoModelForCausalLM.from_pretrained(
                    name, torch_dtype=torch.float32, device_map="cpu"
                ).eval(),
            )
        tokenizer, model = self._hf
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
            )
        return tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def execute(self, requests):
        responses = []

        for request in requests:
            raw = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
            if raw.dtype == np.object_:
                prompts = [p.decode("utf-8") for p in raw.flatten()]
            else:
                prompts = [raw.tobytes().decode("utf-8")]

            generated_texts = []
            for prompt in prompts:
                if isinstance(prompt, bytes):
                    prompt = prompt.decode("utf-8")

                try:
                    generated_texts.append(self._generate_llama_cpp(prompt))
                except Exception as exc:  # llama-server down -> CPU fallback
                    print(f"[generation] llama.cpp unavailable ({exc}); HF fallback", flush=True)
                    generated_texts.append(self._generate_hf_fallback(prompt))

            output_array = np.array(
                [[t.encode("utf-8")] for t in generated_texts], dtype=np.object_
            )
            out_tensor = pb_utils.Tensor("GENERATED_TEXT", output_array)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        pass