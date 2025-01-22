import os
import time
import openai
import json
import datetime
import numpy as np

class VLMPrompter:
    def __init__(self, gpt_version, api_key, root_folder_path, task_name, skill_descriptions=None, plan_execution=None, scene_graph=None, hierarchical_summary=None, images=None, failure_skill=None, failure_reason=None) -> None:
        self.gpt_version = gpt_version
        if not api_key:
            raise ValueError("OpenAI API key is not provided.")
        openai.api_key = api_key

        # Task-specific directory
        self.task_dir = os.path.join(root_folder_path, task_name)
        os.makedirs(self.task_dir, exist_ok=True)

        # Initialize file paths and attributes
        self.skill_descriptions = self.read_file(skill_descriptions) if skill_descriptions else None
        self.plan_execution = self.read_file(plan_execution) if plan_execution else None
        self.scene_graph = self.read_file(scene_graph) if scene_graph else None
        self.hierarchical_summary = self.read_file(hierarchical_summary) if hierarchical_summary else None
        self.images = images if images else []  # List of image file paths
        self.failure_skill = self.read_file(failure_skill) if failure_skill else None
        self.failure_reason = self.read_file(failure_reason) if failure_reason else None

    @staticmethod
    def read_file(file_path):
        """Reads the content of a file and returns it as a string."""
        try:
            with open(file_path, 'r') as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    @staticmethod
    def write_file(file_path, content):
        """Writes content to a file."""
        try:
            with open(file_path, 'w') as file:
                file.write(content.strip())
        except Exception as e:
            print(f"Error writing to file {file_path}: {e}")

    def query(self, prompt: str, sampling_params: dict, save: bool, save_dir: str, query_file: str, response_file: str) -> str:
        """Send the prompt to the GPT model with optional image files and fail-safe retries."""
        # Save query to file
        self.write_file(query_file, prompt)

        # Process images if provided
        image_files = []
        if self.images:
            for image_path in self.images:
                try:
                    with open(image_path, 'rb') as img_file:
                        image_bytes = img_file.read()
                        image_files.append(("image", (os.path.basename(image_path), image_bytes)))
                except Exception as e:
                    print(f"Error: Could not load image '{image_path}'. Exception: {e}")
                    continue

        if image_files and 'gpt-4-vision' not in self.gpt_version:
            raise ValueError("The provided model does not support image input.")

        # Fail-safe mechanism for retries
        max_retries = 5
        retry_count = 0

        while retry_count < max_retries:
            try:
                if image_files:
                    response = openai.ChatCompletion.create(
                        model=self.gpt_version,
                        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
                        files=image_files,
                        **sampling_params
                    )
                else:
                    response = openai.ChatCompletion.create(
                        model=self.gpt_version,
                        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
                        **sampling_params
                    )

                # Successfully received a response
                response_text = response['choices'][0]['message']["content"].strip()

                # Save response to file
                self.write_file(response_file, response_text)

                if save:
                    self.save_response(response, prompt, sampling_params, save_dir)

                return response_text

            except Exception as e:
                retry_count += 1
                print(f"Request failed. Retrying ({retry_count}/{max_retries}) in 2 seconds... Exception: {e}")
                time.sleep(2)

        # If retries exhausted, raise an error
        raise RuntimeError(f"Query failed after {max_retries} retries.")

    def save_response(self, response, prompt, sampling_params, save_dir):
        """Save the GPT response to a file."""
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
                'response': response['choices'][0]['message']["content"].strip(),
                'images': self.images
            }
            json.dump(output, f, indent=4)

    @staticmethod
    def make_key():
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
