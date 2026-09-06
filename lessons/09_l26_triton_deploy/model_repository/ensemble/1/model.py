import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

        self.tokenizer_model = self._get_param("tokenizer_model")
        self.embedding_model = self._get_param("embedding_model")
        self.classification_model = self._get_param("classification_model")
        self.generation_model = self._get_param("generation_model")

    def _get_param(self, name):
        params = self.model_config.get("parameters", {})
        return params.get(name, {}).get("string_value", "")

    def _call_model(self, model_name, inputs):
        tensors = list(inputs.values()) if isinstance(inputs, dict) else inputs
        request = pb_utils.InferenceRequest(
            model_name=model_name,
            inputs=tensors,
            requested_output_names=[],
        )
        response = request.exec()
        if response.has_error():
            err = response.error()
            if hasattr(err, "message"):
                err = err.message()
            raise RuntimeError(f"Error calling {model_name}: {err}")
        return response

    @staticmethod
    def _encode_text(text):
        return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).reshape(1, -1)

    def _tokenize(self, text):
        inputs = {"TEXT": pb_utils.Tensor("TEXT", self._encode_text(text))}
        response = self._call_model(self.tokenizer_model, inputs)

        input_ids = pb_utils.get_output_tensor_by_name(response, "input_ids").as_numpy()
        attention_mask = pb_utils.get_output_tensor_by_name(response, "attention_mask").as_numpy()
        token_type_ids = pb_utils.get_output_tensor_by_name(response, "token_type_ids").as_numpy()
        return input_ids, attention_mask, token_type_ids

    def _mean_pooling(self, hidden_state, attention_mask):
        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = np.sum(hidden_state * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def _get_embedding(self, input_ids, attention_mask, token_type_ids):
        inputs = {
            "input_ids": pb_utils.Tensor("input_ids", input_ids),
            "attention_mask": pb_utils.Tensor("attention_mask", attention_mask),
            "token_type_ids": pb_utils.Tensor("token_type_ids", token_type_ids),
        }
        response = self._call_model(self.embedding_model, inputs)
        hidden_state = pb_utils.get_output_tensor_by_name(response, "last_hidden_state").as_numpy()
        pooled = self._mean_pooling(hidden_state, attention_mask)
        norm = np.linalg.norm(pooled, axis=-1, keepdims=True)
        return pooled / np.clip(norm, a_min=1e-9, a_max=None)

    def _classify(self, input_ids, attention_mask, token_type_ids):
        inputs = {
            "input_ids": pb_utils.Tensor("input_ids", input_ids),
            "attention_mask": pb_utils.Tensor("attention_mask", attention_mask),
            "token_type_ids": pb_utils.Tensor("token_type_ids", token_type_ids),
        }
        response = self._call_model(self.classification_model, inputs)
        logits = pb_utils.get_output_tensor_by_name(response, "logits").as_numpy().astype(np.float64)
        z = logits - np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(z)
        probs /= probs.sum(axis=-1, keepdims=True)
        label = int(np.argmax(probs, axis=-1)[0])
        confidence = float(probs.max())
        return label, confidence

    def _generate(self, text):
        inputs = {"TEXT": pb_utils.Tensor("TEXT", self._encode_text(text))}
        response = self._call_model(self.generation_model, inputs)
        output = pb_utils.get_output_tensor_by_name(response, "GENERATED_TEXT")
        return output.as_numpy()[0][0]

    def execute(self, requests):
        responses = []

        for request in requests:
            raw = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
            if raw.dtype == np.object_:
                text = raw.flatten()[0]
            else:
                text = raw.tobytes().decode("utf-8")

            if isinstance(text, bytes):
                text = text.decode("utf-8")

            input_ids, attention_mask, token_type_ids = self._tokenize(text)
            embedding = self._get_embedding(input_ids, attention_mask, token_type_ids)
            label, confidence = self._classify(input_ids, attention_mask, token_type_ids)

            if label == 1:
                generated = self._generate(text)
                if isinstance(generated, bytes):
                    generated = generated.decode("utf-8")
                response_text = f"[Generated] {generated}"
            else:
                response_text = (
                    f"[Classification] label={label}, confidence={confidence:.3f}"
                )

            output_array = np.array(
                [[response_text.encode("utf-8")]], dtype=np.object_
            )
            out_tensor = pb_utils.Tensor("RESPONSE", output_array)
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_tensor])
            )

        return responses

    def finalize(self):
        pass
