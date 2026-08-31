import os
import time
import openai
import json
import datetime
import numpy as np
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class VLMPrompter:

    def __init__(
        self,
        gpt_version="gpt-4o",
        api_key=None,
        root_folder_path=None,
        task_name=None,
        skill_descriptions=None,
        plan_execution=None,
        scene_graph="scene_graph.txt",
        execution_history="execution_history.txt",
        images=None,
        failure_skill=None,
        failure_reason=None,
        resources=None,
        verbose=True,
        vlm_run=True,
    ) -> None:

        if not api_key:
            raise ValueError("OpenAI API key is not provided.")
        openai.api_key = api_key
        self.root_folder_path = root_folder_path

        self.gpt_version = gpt_version
        self.verbose = verbose
        self.proactive = True
        self.vlm_run = vlm_run
        self.img_failure = True
        self.sg = True
        self.history = True
        self.no_images = 2

        # Task-specific directory
        self.task_dir = os.path.join(root_folder_path, task_name)
        os.makedirs(self.task_dir, exist_ok=True)

        # Reset hierarchical summary
        with open(os.path.join(self.task_dir, execution_history), 'w', encoding='utf-8') as f:
            f.write("")
        with open(os.path.join(self.task_dir, "failure_skill.txt"), 'w', encoding='utf-8') as f:
            f.write("")
        with open(os.path.join(self.task_dir, "failure_reason.txt"), 'w', encoding='utf-8') as f:
            f.write("")

        # Load prompts JSON file
        self.resources = resources
        self.prompts_json_file = self.read_json_file(os.path.join(resources, "prompts_real.json"))
        self.images = images if images else []  # List of image file paths
        if not self.prompts_json_file:
            raise ValueError("Invalid or missing prompts JSON file.")

        # variables
        self.skill_name = False
        self.skill_preconditions = False
        self.skill_postconditions = False

    def get_files(self):
        # Initialize file paths and attributes
        file = os.path.join(self.resources, "skill_descriptions.json")
        self.skill_descriptions = self.read_json_file(file) if os.path.exists(file) else None

        files = ["plan", "scene_graph", "execution_history", "failure_skill", "failure_reason"]        
        self.plan_execution, self.scene_graph, self.execution_history, \
            self.failure_skill, self.failure_reason = [self.read_file(os.path.join(self.resources, f + ".txt")) if os.path.exists(f) else None for f in files]

    @staticmethod
    def read_json_file(file_path):
        """
        Reads the content of a JSON file and returns it as a Python dictionary.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {file_path}: {e}")
            return None
        except Exception as e:
            print(f"Error reading JSON file {file_path}: {e}")
            return None
    @staticmethod
    def read_file(file_path):
        """Reads the content of a file and returns it as a string."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    @staticmethod
    def write_file(file_path, content):
        """Writes content to a file."""
        try:
            if type(content) == dict:
                with open(file_path.replace('txt', json), 'w') as file:
                    json.dump(content, file, indent=4)
            else:
                with open(file_path, 'w') as file:
                    file.write(content.strip())
        except Exception as e:
            print(f"Error writing to file {file_path}: {e}")

    def proactive_check(self):
        """
        Proactive check using detection, identification, and correction.
        """
        # Update dynamic inputs
        self.update_inputs()

        # Detection
        self.proactive = False
        self.check_type = "proactive_detection"
        detection_result = self.proactive_detection()

        if "No" in detection_result:
            # Identification
            self.check_type = "proactive_identification"
            identification_result = self.proactive_identification()

            # Correction
            self.check_type = "proactive_correction"
            correction_result = self.proactive_correction()

            if self.verbose:
                print(f"Proactive Check - Detection: {detection_result}")
                print(f"Proactive Check - Identification: {identification_result}")
                print(f"Proactive Check - Correction: {correction_result}")

            return False  # Preconditions not satisfied after corrections

        return True  # Preconditions satisfied

    def precondition_verifier_check(self):
        """
        Check preconditions using detection, identification, and correction.
        """
        # Update dynamic inputs
        self.update_inputs()

        # Detection
        self.check_type = "precondition_verifier_detection"
        detection_result = self.precondition_verifier_detection()

        if "No" in detection_result:
            # Identification
            self.check_type = "precondition_verifier_identification"
            identification_result = self.precondition_verifier_identification()

            # Correction
            self.check_type = "precondition_verifier_correction"
            correction_result = self.precondition_verifier_correction()

            if self.verbose:
                print(f"Precondition Verifier Check - Detection: {detection_result}")
                print(f"Precondition Verifier Check - Identification: {identification_result}")
                print(f"Precondition Verifier Check - Correction: {correction_result}")

            feiled_condition = correction_result
            return False #feiled_condition # Preconditions not satisfied after corrections

        return True  # Preconditions satisfied

    def precondition_suggestor_check(self):
        """
        Suggests new preconditions using detection, identification, and correction.
        """
        # Update dynamic inputs
        self.update_inputs()

        # Detection
        self.check_type = "precondition_suggestor_detection"
        detection_result = self.precondition_suggestor_detection()

        if "Missing preconditions detected" in detection_result:
            # Identification
            self.check_type = "precondition_suggestor_identification"
            identification_result = self.precondition_suggestor_identification()

            # Correction
            self.check_type = "precondition_suggestor_correction"
            correction_result = self.precondition_suggestor_correction()

            if self.verbose:
                print(f"Precondition Suggestor Check - Detection: {detection_result}")
                print(f"Precondition Suggestor Check - Identification: {identification_result}")
                print(f"Precondition Suggestor Check - Correction: {correction_result}")

            return False  # New preconditions suggested

        return True  # No new preconditions are required

    def postcondition_verifier_check(self):
        """
        Check postconditions using detection, identification, and correction.
        """
        # # Update dynamic inputs
        self.update_inputs()

        # Detection
        self.check_type = "postcondition_verifier_detection"
        detection_result = self.postcondition_verifier_detection()

        if "No" in detection_result:
            # Identification
            self.check_type = "postcondition_verifier_identification"
            identification_result = self.postcondition_verifier_identification()

            # Correction
            self.check_type = "postcondition_verifier_correction"
            correction_result = self.postcondition_verifier_correction()

            if self.verbose:
                print(f"Postcondition Verifier Check - Detection: {detection_result}")
                print(f"Postcondition Verifier Check - Identification: {identification_result}")
                print(f"Postcondition Verifier Check - Correction: {correction_result}")

            failed_postcondition = correction_result
            # return failed_postcondition  # Postconditions not satisfied after corrections
            return False

        return True  # Postconditions satisfied

    def postcondition_suggestor_check(self):
        """
        Suggests new postconditions using detection, identification, and correction.
        """
        # Update dynamic inputs
        self.update_inputs()

        # Detection
        self.check_type = "postcondition_suggestor_detection"
        detection_result = self.postcondition_suggestor_detection()

        if "No" in detection_result:
            # Identification
            self.check_type = "postcondition_suggestor_identification"
            identification_result = self.postcondition_suggestor_identification()

            # Correction
            self.check_type = "postcondition_suggestor_correction"
            correction_result = self.postcondition_suggestor_correction()

            if self.verbose:
                print(f"Postcondition Suggestor Check - Detection: {detection_result}")
                print(f"Postcondition Suggestor Check - Identification: {identification_result}")
                print(f"Postcondition Suggestor Check - Correction: {correction_result}")

            return None #False  # New postconditions suggested

        return None #True  # No new postconditions are required

    def skill_suggestor_check(self):
        """
        Suggests new skills.
        """
        # Update dynamic inputs
        self.update_inputs()

        # Detection
        self.check_type = "skill_suggestor_detection"
        detection_result = self.skill_suggestor_detection()

        if not "No skill needed" in detection_result:
            if self.verbose:
                print(f"skill Suggestor Check - Detection: {detection_result}")
            # Identification
            # self.check_type = "skill_generator"
            # skill = self.skill_generator()
            # with open(os.path.join(self.task_dir, "skill.py"), 'w') as f:
            #     f.write(skill)

            return False #False  # New skill suggested

        return True #True  # No new skill are required

    def update_inputs(self, images=None, scene_graph="scene_graph.txt", execution_history="execution_history.txt"):
        """Updates dynamic inputs like images, scene graph, and hierarchical summary."""
        # images = [os.path.join("BETR-XP-LLM/detections", 'rgb.jpg')]
        images = [img for img in os.listdir(self.task_dir) if img.startswith("image")] if self.img_failure \
            else [img for img in os.listdir(self.task_dir) if img.startswith("no_failure")]
        images = [os.path.join(self.task_dir, img) for img in images]
        file = os.path.join(self.resources, "skill_descriptions.json")
        self.skill_descriptions = self.read_json_file(file) if os.path.exists(file) else None

        if images:
            self.images = [img for img in images if os.path.exists(img)]

        files = ["plan.txt", scene_graph, execution_history, "failure_skill.txt", "failure_reason.txt", "task_description.txt"]
        self.plan_execution, self.scene_graph, self.execution_history, \
            self.failure_skill, self.failure_reason, self.task_description = [self.read_file(os.path.join(self.task_dir, f)) if os.path.exists(os.path.join(self.task_dir, f)) else None for f in files]
            
        if not self.sg:
            self.execution_history = self.execution_history.split("\n")
            self.execution_history = [line for line in self.execution_history if "Observation:" not in line]
            self.execution_history = "\n".join(self.execution_history)
        
        if not self.proactive:
            if self.skill_name == False:
                self.skill_name = self.execution_history.split("|")[-2].split(":")[-1]
            if self.skill_preconditions == False:
                self.skill_preconditions = self.execution_history.split("|")[-1].split(":")[-1]
            if self.skill_postconditions == False:
                self.skill_postconditions = self.execution_history.split("|")[-1].split(":")[-1]

    def extract_failure_skill(self, response):
        """Extracts the failure skill from the GPT response."""
        # Use a simple parsing logic or a regex to identify the skill name in the response
        try:
            # Example: Extract the failure skill from the response text
            # Assuming response is structured like: "Root cause of failure is [SKILL] because [REASON]"
            skill_start = response.find("Root cause of failure is") + len("Root cause of failure is")
            skill_end = response.find("because")
            failure_skill = response[skill_start:skill_end].strip()
            return failure_skill
        except Exception as e:
            print(f"Error extracting failure skill: {e}")
            return None

    def extract_failure_reason(self, response):
        """Extracts the failure reason from the GPT response."""
        try:
            # Example: Extract the reason from the response text
            # Assuming response is structured like: "Root cause of failure is [SKILL] because [REASON]"
            reason_start = response.find("because") + len("because")
            failure_reason = response[reason_start:].strip()
            return failure_reason
        except Exception as e:
            print(f"Error extracting failure reason: {e}")
            return None

    def query(self, prompt: str, sampling_params: dict, save: bool, save_dir: str, query_file: str, response_file: str) -> str:
        """Send the prompt to the GPT model with optional image files and fail-safe retries."""
        # Save query to file
        self.write_file(query_file, prompt['user'])

        # Process images if provided
        image_files = []
        if self.images:
            for image_path in self.images:
                try:
                    base64_image = encode_image(image_path)
                    img_prompt = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high",},}
                    image_files.append(img_prompt)
                except Exception as e:
                    print(f"Error: Could not load image '{image_path}'. Exception: {e}")
                    continue

        if self.no_images and len(image_files) > self.no_images:
            image_files = image_files[:self.no_images]

        # if image_files and 'gpt-4o-mini' not in self.gpt_version:
        #     raise ValueError("The provided model does not support image input.")

        # Fail-safe mechanism for retries
        max_retries = 5
        retry_count = 0
        content = [{"type": "text", "text": prompt['user']}]
        for img in image_files:
            content.append(img)

        while retry_count < max_retries:
            try:
                if image_files:
                    response = openai.ChatCompletion.create(
                        model=self.gpt_version,
                        messages=[
                            {"role": "system", "content": prompt['system']},
                            {"role": "user", "content": content},
                            ],
                        **sampling_params
                    )

                else:
                    response = openai.ChatCompletion.create(
                        model=self.gpt_version,
                        messages=[{"role": "system", "content": prompt['system']}, {"role": "user", "content": prompt['user']}],
                        **sampling_params
                    )

                # Successfully received a response
                response_text = response['choices'][0]['message']["content"].strip()
                if self.verbose:
                    print(f"Response: {response_text}")

                # Save response to file
                self.write_file(response_file, response_text)

                if False: #save:
                    self.save_response(response, prompt, sampling_params, save_dir)
                    restult= {
                        "prompt": prompt,
                        "response": response_text,
                    }
                    os.makedirs(f"{self.save_dir}", exist_ok=True)
                    with open(os.path.join(f"{self.save_dir}", f"{self.check_type}_.json"), 'w') as f:
                        json.dump(restult, f, indent=4)

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
            with open(response_file, "r", encoding="utf-8") as f:
                output = json.load(f)

        with open(response_file, "w", encoding="utf-8") as f:
            output[key] = {
                'prompt': prompt,
                'sampling_params': sampling_params,
                'response': response['choices'][0]['message']["content"].strip(),
                'images': self.images
            }
            json.dump(output, f, indent=4)

    @staticmethod
    def make_key():
        """Generate a unique key based on the current date and time."""
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def _populate_prompt(self, params, include_failure_info=False):
        """Populates placeholders in the prompt with actual data."""
        prompt = {}
        user_prompt = params["template-user"]

        user_prompt = user_prompt.replace("[TASK-DESCRIPTION]", self.task_description or "")
        user_prompt = user_prompt.replace("[SKILL-NAME]", self.skill_name or "")
        user_prompt = user_prompt.replace("[SKILL-PRECONDITIONS]", self.skill_preconditions or "")
        user_prompt = user_prompt.replace("[SKILL-POSTCONDITIONS]", self.skill_postconditions or "")
        user_prompt = user_prompt.replace("[SKILL-DESCRIPTIONS]", f"{self.skill_descriptions['skills']}" or "")
        user_prompt = user_prompt.replace("[CONDITION-DESCRIPTIONS]", f"{self.skill_descriptions['conditions']}" or "")
        user_prompt = user_prompt.replace("[PLAN-EXECUTION]", self.plan_execution or "")
        if self.sg:
            user_prompt = user_prompt.replace("[SCENE-GRAPH]", self.scene_graph or "")
        else:
            user_prompt = user_prompt.replace("[SCENE-GRAPH]", "Scene graph not available. Use the image for reference. Image is the ground truth. Analyze the image to identify spatial relationships.")
        user_prompt = user_prompt.replace("[KNOWN-OBJECT-LOCATIONS]", "" or "")
        if self.history:
            user_prompt = user_prompt.replace("[HIERARCHICAL-SUMMARY]", self.execution_history or "")
        else:
            user_prompt = user_prompt.replace("[HIERARCHICAL-SUMMARY]", "No hierarchical summary available. Use the image and plan execution for reference.")
        user_prompt = user_prompt.replace("[IMAGES]", ", ".join(self.images) if self.images else "")

        if include_failure_info:
            user_prompt = user_prompt.replace("[FAILURE-SKILL]", self.failure_skill or "")
            user_prompt = user_prompt.replace("[FAILURE-REASON]", self.failure_reason or "")
        else:
            user_prompt = user_prompt.replace("[FAILURE-SKILL]", "")
            user_prompt = user_prompt.replace("[FAILURE-REASON]", "")

        prompt["user"] = user_prompt
        prompt["system"] = params["template-system"]

        return prompt

    # Precondition methods
    def precondition_verifier_detection(self):
        """Handles the verifier detection functionality for preconditions."""

        params = self.prompts_json_file["preconditionverifier"]["template-detection"]

        prompt = self._populate_prompt(params, include_failure_info=False)
        query_file = os.path.join(self.task_dir, "preconditions_detection_query.txt")
        response_file = os.path.join(self.task_dir, "preconditions_detection_response.txt")
        return self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

    def precondition_suggestor_detection(self):
        """Handles the suggestor detection functionality for preconditions."""

        params = self.prompts_json_file["preconditionsuggestor"]["template-detection"]

        prompt = self._populate_prompt(params, include_failure_info=False)
        query_file = os.path.join(self.task_dir, "suggestor_detection_query.txt")
        response_file = os.path.join(self.task_dir, "suggestor_detection_response.txt")
        return self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

    def precondition_verifier_identification(self):
        """Handles the verifier identification functionality for preconditions."""

        params = self.prompts_json_file["preconditionverifier"]["template-identification"]

        prompt = self._populate_prompt(params, include_failure_info=False)
        query_file = os.path.join(self.task_dir, "preconditions_identification_query.txt")
        response_file = os.path.join(self.task_dir, "preconditions_identification_response.txt")
        response = self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

        failure_skill = self.extract_failure_skill(response)
        failure_reason = self.extract_failure_reason(response)

        if failure_skill:
            self.write_file(os.path.join(self.task_dir, "failure_skill.txt"), failure_skill)
        else:
            print("No failure skill identified.")

        if failure_reason:
            self.write_file(os.path.join(self.task_dir, "failure_reason.txt"), failure_reason)
        else:
            print("No failure reason identified.")

        return response

    def precondition_suggestor_identification(self):
        """Handles the suggestor identification functionality for preconditions."""

        params = self.prompts_json_file["preconditionsuggestor"]["template-identification"]

        prompt = self._populate_prompt(params, include_failure_info=True)
        query_file = os.path.join(self.task_dir, "suggestor_identification_query.txt")
        response_file = os.path.join(self.task_dir, "suggestor_identification_response.txt")
        response = self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

        failure_skill = self.extract_failure_skill(response)
        failure_reason = self.extract_failure_reason(response)

        if failure_skill:
            self.write_file(os.path.join(self.task_dir, "failure_skill.txt"), failure_skill)
        else:
            print("No failure skill identified.")

        if failure_reason:
            self.write_file(os.path.join(self.task_dir, "failure_reason.txt"), failure_reason)
        else:
            print("No failure reason identified.")

        return response

    def precondition_verifier_correction(self):
        """Handles the verifier correction functionality for preconditions."""

        params = self.prompts_json_file["preconditionverifier"]["template-correction"]
        failure_skill = self.read_file(os.path.join(self.task_dir, "failure_skill.txt"))
        failure_reason = self.read_file(os.path.join(self.task_dir, "failure_reason.txt"))
        if not failure_skill or not failure_reason:
            raise ValueError("Missing failure skill or reason. Ensure identification is run first.")

        self.failure_skill = failure_skill
        self.failure_reason = failure_reason

        prompt = self._populate_prompt(params, include_failure_info=True)
        query_file = os.path.join(self.task_dir, "preconditions_correction_query.txt")
        response_file = os.path.join(self.task_dir, "preconditions_correction_response.txt")
        return self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

    def precondition_suggestor_correction(self):
        """Handles the suggestor correction functionality for preconditions."""

        params = self.prompts_json_file["preconditionsuggestor"]["template-correction"]
        failure_skill = self.read_file(os.path.join(self.task_dir, "failure_skill.txt"))
        failure_reason = self.read_file(os.path.join(self.task_dir, "failure_reason.txt"))
        if not failure_skill or not failure_reason:
            raise ValueError("Missing failure skill or reason. Ensure identification is run first.")

        self.failure_skill = failure_skill
        self.failure_reason = failure_reason

        prompt = self._populate_prompt(params, include_failure_info=True)
        query_file = os.path.join(self.task_dir, "suggestor_correction_query.txt")
        response_file = os.path.join(self.task_dir, "suggestor_correction_response.txt")
        return self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

    # Postcondition methods
    def postcondition_verifier_detection(self):
        """Handles the verifier detection functionality for postconditions."""

        params = self.prompts_json_file["postconditionverifier"]["template-detection"]

        prompt = self._populate_prompt(params, include_failure_info=False)
        query_file = os.path.join(self.task_dir, "postconditions_detection_query.txt")
        response_file = os.path.join(self.task_dir, "postconditions_detection_response.txt")
        return self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

    def postcondition_suggestor_detection(self):
        """Handles the suggestor detection functionality for postconditions."""

        params = self.prompts_json_file["postconditionsuggestor"]["template-detection"]

        prompt = self._populate_prompt(params, include_failure_info=False)
        query_file = os.path.join(self.task_dir, "postconditions_detection_query.txt")
        response_file = os.path.join(self.task_dir, "postconditions_detection_response.txt")
        return self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

    def postcondition_verifier_identification(self):
        """Handles the verifier identification functionality for postconditions."""

        params = self.prompts_json_file["postconditionverifier"]["template-identification"]

        prompt = self._populate_prompt(params, include_failure_info=False)
        query_file = os.path.join(self.task_dir, "postconditions_identification_query.txt")
        response_file = os.path.join(self.task_dir, "postconditions_identification_response.txt")
        response = self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

        failure_skill = self.extract_failure_skill(response)
        failure_reason = self.extract_failure_reason(response)

        if failure_skill:
            self.write_file(os.path.join(self.task_dir, "failure_skill.txt"), failure_skill)
        else:
            print("No failure skill identified.")

        if failure_reason:
            self.write_file(os.path.join(self.task_dir, "failure_reason.txt"), failure_reason)
        else:
            print("No failure reason identified.")
        return response

    def postcondition_suggestor_identification(self):
        """Handles the suggestor identification functionality for postconditions."""

        params = self.prompts_json_file["postconditionsuggestor"]["template-identification"]

        prompt = self._populate_prompt(params, include_failure_info=True)
        query_file = os.path.join(self.task_dir, "postconditions_identification_query.txt")
        response_file = os.path.join(self.task_dir, "postconditions_identification_response.txt")
        response = self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

        failure_skill = self.extract_failure_skill(response)
        failure_reason = self.extract_failure_reason(response)

        if failure_skill:
            self.write_file(os.path.join(self.task_dir, "failure_skill.txt"), failure_skill)
        else:
            print("No failure skill identified.")

        if failure_reason:
            self.write_file(os.path.join(self.task_dir, "failure_reason.txt"), failure_reason)
        else:
            print("No failure reason identified.")
        return response

    def postcondition_verifier_correction(self):
        """Handles the verifier correction functionality for postconditions."""

        params = self.prompts_json_file["postconditionverifier"]["template-correction"]
        failure_skill = self.read_file(os.path.join(self.task_dir, "failure_skill.txt"))
        failure_reason = self.read_file(os.path.join(self.task_dir, "failure_reason.txt"))
        if not failure_skill or not failure_reason:
            raise ValueError("Missing failure skill or reason. Ensure identification is run first.")

        self.failure_skill = failure_skill
        self.failure_reason = failure_reason

        prompt = self._populate_prompt(params, include_failure_info=True)
        query_file = os.path.join(self.task_dir, "postconditions_correction_query.txt")
        response_file = os.path.join(self.task_dir, "postconditions_correction_response.txt")
        return self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

    def postcondition_suggestor_correction(self):
        """Handles the suggestor correction functionality for postconditions."""

        params = self.prompts_json_file["postconditionsuggestor"]["template-correction"]
        failure_skill = self.read_file(os.path.join(self.task_dir, "failure_skill.txt"))
        failure_reason = self.read_file(os.path.join(self.task_dir, "failure_reason.txt"))
        if not failure_skill or not failure_reason:
            raise ValueError("Missing failure skill or reason. Ensure identification is run first.")

        self.failure_skill = failure_skill
        self.failure_reason = failure_reason

        prompt = self._populate_prompt(params, include_failure_info=True)
        query_file = os.path.join(self.task_dir, "postconditions_correction_query.txt")
        response_file = os.path.join(self.task_dir, "postconditions_correction_response.txt")
        return self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

    # Proactive Checker methods (static inputs)
    def proactive_detection(self):
        """Handles the detection functionality for proactive checking."""

        params = self.prompts_json_file["proactivechecker"]["template-detection"]

        prompt = self._populate_prompt(params, include_failure_info=False)
        query_file = os.path.join(self.task_dir, "proactive_detection_query.txt")
        response_file = os.path.join(self.task_dir, "proactive_detection_response.txt")
        return self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

    def proactive_identification(self):
        """Handles the identification functionality for proactive checking."""

        params = self.prompts_json_file["proactivechecker"]["template-identification"]

        prompt = self._populate_prompt(params, include_failure_info=True)
        query_file = os.path.join(self.task_dir, "proactive_identification_query.txt")
        response_file = os.path.join(self.task_dir, "proactive_identification_response.txt")
        response = self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

        failure_skill = self.extract_failure_skill(response)
        failure_reason = self.extract_failure_reason(response)

        if failure_skill:
            self.write_file(os.path.join(self.task_dir, "failure_skill.txt"), failure_skill)
        else:
            print("No failure skill identified.")

        if failure_reason:
            self.write_file(os.path.join(self.task_dir, "failure_reason.txt"), failure_reason)
        else:
            print("No failure reason identified.")
        return response

    def proactive_correction(self):
        """Handles the correction functionality for proactive checking."""

        params = self.prompts_json_file["proactivechecker"]["template-correction"]
        failure_skill = self.read_file(os.path.join(self.task_dir, "failure_skill.txt"))
        failure_reason = self.read_file(os.path.join(self.task_dir, "failure_reason.txt"))
        if not failure_skill or not failure_reason:
            raise ValueError("Missing failure skill or reason. Ensure identification is run first.")

        self.failure_skill = failure_skill
        self.failure_reason = failure_reason

        prompt = self._populate_prompt(params, include_failure_info=True)
        query_file = os.path.join(self.task_dir, "proactive_correction_query.txt")
        response_file = os.path.join(self.task_dir, "proactive_correction_response.txt")
        return self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

    def skill_suggestor_detection(self):
        """Handles the suggestor detection functionality for postconditions."""

        params = self.prompts_json_file["skillsuggestor"]["template-detection"]

        prompt = self._populate_prompt(params, include_failure_info=False)
        query_file = os.path.join(self.task_dir, "skill_suggestor_query.txt")
        response_file = os.path.join(self.task_dir, "skill_suggestor_response.txt")
        return self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

    def skill_generator(self):
        """Handles the suggestor detection functionality for postconditions."""

        params = self.prompts_json_file["skillsuggestor"]["skill-generator"]

        prompt = self._populate_prompt(params, include_failure_info=False)
        query_file = os.path.join(self.task_dir, "skill_suggestor_query.txt")
        response_file = os.path.join(self.task_dir, "skill_suggestor_response.txt")
        return self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)