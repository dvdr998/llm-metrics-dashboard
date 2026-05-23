import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from services.device_metrics import get_best_device, get_device_snapshot, reset_peak_memory


class LocalLLMRunner:
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.device = get_best_device()
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """
        Loads the local Hugging Face model.
        For now, we use distilgpt2 because it is small and beginner-friendly.
        """
        start_time = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        if self.device in ["cuda", "mps"]:
            self.model = self.model.to(self.device)

        self.model.eval()

        end_time = time.perf_counter()

        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_load_time_sec": round(end_time - start_time, 4)
        }

    def generate(self, prompt: str, max_new_tokens: int = 80):
        """
        Runs local LLM inference and collects performance metrics.
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()

        reset_peak_memory()

        before_snapshot = get_device_snapshot()

        inputs = self.tokenizer(prompt, return_tensors="pt")

        if self.device in ["cuda", "mps"]:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

        input_token_count = inputs["input_ids"].shape[-1]

        start_time = time.perf_counter()

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

        end_time = time.perf_counter()

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        total_tokens = output_ids.shape[-1]
        output_token_count = total_tokens - input_token_count

        latency_sec = end_time - start_time
        tokens_per_second = output_token_count / latency_sec if latency_sec > 0 else 0

        after_snapshot = get_device_snapshot()

        return {
            "response": generated_text,
            "model_name": self.model_name,
            "device": self.device,
            "input_tokens": int(input_token_count),
            "output_tokens": int(output_token_count),
            "total_tokens": int(total_tokens),
            "latency_sec": round(latency_sec, 4),
            "tokens_per_second": round(tokens_per_second, 4),
            "before_snapshot": before_snapshot,
            "after_snapshot": after_snapshot
        }
