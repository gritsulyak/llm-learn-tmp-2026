import os
import numpy as np
import transformers
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args: dict) -> None:
        model_dir = os.path.dirname(os.path.abspath(__file__))
        local_tokenizer_dir = os.path.join(model_dir, "tokenizer")

        model_name = "huawei-noah/TinyBERT_General_4L_312D"

        if os.path.isdir(local_tokenizer_dir):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                local_tokenizer_dir, local_files_only=True
            )
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        self.max_length = 128

    def execute(self, requests: list) -> list:
        responses = []
        for request in requests:
            raw = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
            if raw.dtype == np.object_:
                texts = [t.decode("utf-8") for t in raw.flatten()]
            else:
                texts = [raw.tobytes().decode("utf-8")]

            encoded = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )

            input_ids = encoded["input_ids"].astype(np.int64)
            attention_mask = encoded["attention_mask"].astype(np.int64)
            token_type_ids = encoded["token_type_ids"].astype(np.int64)
            print(f"[debug] tokenizer texts={texts!r} ids.shape={input_ids.shape} size={input_ids.size}", flush=True)

            out_tensors = [
                pb_utils.Tensor("input_ids", input_ids),
                pb_utils.Tensor("attention_mask", attention_mask),
                pb_utils.Tensor("token_type_ids", token_type_ids),
            ]

            responses.append(pb_utils.InferenceResponse(output_tensors=out_tensors))
        return responses

    def finalize(self):
        pass
