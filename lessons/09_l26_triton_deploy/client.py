"""Simple Triton client checking the full BLS pipeline with one request.

Usage:
    uv run python client.py "What is machine learning?"
    uv run python client.py --host 192.168.1.5 --port 8001 "Hello world"
"""

import argparse
import sys

import numpy as np
import tritonclient.grpc as grpcclient


def make_text_input(name, text):
    arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).reshape(1, -1)
    infer_input = grpcclient.InferInput(name, list(arr.shape), "UINT8")
    infer_input.set_data_from_numpy(arr)
    return infer_input


def main():
    parser = argparse.ArgumentParser(description="Triton BLS pipeline client")
    parser.add_argument("--host", default="localhost", help="Triton gRPC host")
    parser.add_argument("--port", type=int, default=8001, help="Triton gRPC port")
    parser.add_argument("--model", default="ensemble", help="Model name to query")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("texts", nargs="+", help="input text(s) to send")
    args = parser.parse_args()

    url = f"{args.host}:{args.port}"
    client = grpcclient.InferenceServerClient(url=url, verbose=False)

    if not client.is_server_live():
        sys.exit(f"ERROR: Triton server not live at {url}")

    for text in args.texts:
        inputs = [make_text_input("TEXT", text)]
        outputs = [grpcclient.InferRequestedOutput("RESPONSE")]
        result = client.infer(args.model, inputs, outputs=outputs, timeout=args.timeout)

        responses = result.as_numpy("RESPONSE")
        raw = responses[0][0]
        answer = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        print(f"{text}\n  -> {answer}\n")


if __name__ == "__main__":
    main()