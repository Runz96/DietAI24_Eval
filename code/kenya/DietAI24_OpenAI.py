import os
import base64
import json
from dotenv import load_dotenv
from openai import OpenAI

class DietAI24:
    def __init__(self, model_name, food_list_path="../data/kenya/kenya_food_list.txt"):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model_name

        # Load food list and build a general-purpose system prompt
        kenyan_foods = self.load_food_list(food_list_path)
        system_prompt = (
            "You are a Kenyan food assistant. You can recognize Kenyan foods from images and estimate the weight "
            "of food portions in grams. For each task, always provide your output in the requested JSON format. "
            "If you are unsure, respond with null for food name or weight as appropriate. "
            "Here is a list of common Kenyan foods you know about:\n- " +
            "\n- ".join(kenyan_foods)
        )
        self.messages = [{"role": "system", "content": system_prompt}]

    def load_food_list(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    def add_user_message(self, text, image_path=None):
        content = [{"type": "text", "text": text}]
        if image_path:
            ext = os.path.splitext(image_path)[-1].lower()
            mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
            image_base64 = self.encode_image_to_base64(image_path)
            # NEW: request higher image detail
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{image_base64}",
                    "detail": "high"  # <-- added
                }
            })
        self.messages.append({"role": "user", "content": content})
        
    def get_response(self, max_tokens=500):
        # NEW: temperature=0 and JSON-only response_format
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=max_tokens,
            temperature=0,
            response_format={"type": "json_object"}
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def reset_conversation(self):
        # Keep only system prompt
        self.messages = self.messages[:1]

    def recognize_food(self, image_path):
        self.add_user_message(
            (
                "Please identify the most relevant Kenyan food shown in the image."
                "First, try to match the food to one from the provided list of common Kenyan foods (see system prompt)."
                "If you cannot find a match from the list, you may use your own knowledge of Kenyan cuisine."
                "Provide your answer in the following JSON format with exactly these keys:\n"
                '{\n'
                '  "food_name": string|null,\n'
                '  "reasoning": string,\n'
                '  "confidence": number  // between 0 and 1\n'
                '}\n'
                "Only a single food name. Do not list multiple food names. Use null if unsure."
            ),
            image_path=image_path
        )
        return self.get_response()

    def estimate_weight(self, assumed_food=None, image_path=None):
        prompt = (
            f"Estimate the weight (in grams) of {assumed_food}."
            "Use photographic cues (plate/utensils/hand/container/perspective) and typical portion sizes."
            "Provide your answer in the following JSON format with exactly these keys:\n"
            '{\n'
            '  "weight_grams": number|null,\n'
            '  "reasoning": string,\n'
            '  "confidence": number  // between 0 and 1\n'
            '}\n'
            "Only a single numeric value for weight_grams (no ranges). Use null if truly unsure."
        )
        self.add_user_message(prompt, image_path=image_path)
        return self.get_response()