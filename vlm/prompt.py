import os
import time
import openai
import json
import datetime
import numpy as np

class VLMPrompter:
    def __init__(self, gpt_version, api_key) -> None:
        self.gpt_version = gpt_version
        if not api_key:
            raise ValueError("OpenAI API key is not provided.")
        openai.api_key = api_key

    def query(self, prompt: str, sampling_params: dict, save: bool, save_dir: str, image_paths: list = None) -> str:
        # Process images if provided
        image_files = []
        if image_paths:
            for image_path in image_paths:
                try:
                    with open(image_path, 'rb') as img_file:
                        image_bytes = img_file.read()
                        image_files.append(("image", (os.path.basename(image_path), image_bytes)))
                except Exception as e:
                    print(f"Error: Could not load image '{image_path}'. Exception: {e}")
                    continue

        if image_files and 'gpt-4-vision' not in self.gpt_version:
            raise ValueError("The provided model does not support image input.")

        while True:
            try:
                # Handle multimodal input or text-only input
                if image_files:
                    response = openai.ChatCompletion.create(
                        model=self.gpt_version,
                        messages=[
                            {"role": "system", "content": prompt['system']},
                            {"role": "user", "content": prompt['user']},
                        ],
                        files=image_files,
                        **sampling_params
                    )
                else:
                    response = openai.ChatCompletion.create(
                        model=self.gpt_version,
                        messages=[
                            {"role": "system", "content": prompt['system']},
                            {"role": "user", "content": prompt['user']},
                        ],
                        **sampling_params
                    )
            except Exception as e:
                print(f"Request failed. Retrying in 2 seconds... Exception: {e}")
                time.sleep(2)
                continue
            break

        if save:
            self.save_response(response, prompt, sampling_params, save_dir, image_paths)

        if 'gpt-4' in self.gpt_version:
            return response['choices'][0]['message']["content"].strip(), None
        else:
            logprob = response['choices'][0].get('logprobs', {}).get('token_logprobs')
            return response['choices'][0]['text'].strip(), np.mean(logprob) if logprob else None

    def save_response(self, response, prompt, sampling_params, save_dir, image_paths):
        os.makedirs(save_dir, exist_ok=True)
        key = self.make_key()
        output = {}

        response_file = os.path.join(save_dir, 'response.json')
        if os.path.exists(response_file):
            with open(response_file, 'r') as f:
                output = json.load(f)

        with open(response_file, 'w') as f:
            output[key] = {
                'prompt': prompt,
                'sampling_params': sampling_params,
                'response': response['choices'][0]['message']["content"].strip() if 'gpt-4' in self.gpt_version else response['choices'][0]['text'].strip(),
                'images': image_paths
            }
            json.dump(output, f, indent=4)

    @staticmethod
    def make_key():
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
