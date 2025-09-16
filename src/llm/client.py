import base64
from typing import List, Optional, Dict, Any

from openai import OpenAI
from src.config.settings import settings

class LLMClient:
    """
    A wrapper for the OpenAI-compatible Large Language Model API.

    Handles client initialization, prompt construction, and API calls,
    including support for multi-modal (image) inputs and JSON mode.
    """
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url
        )
        self.model = settings.llm.model_name

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encodes an image file to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[List[str]] = None,
        json_mode: bool = False
    ) -> str:
        """
        Generates a response from the LLM.

        Args:
            system_prompt: The system-level instructions for the model.
            user_prompt: The user's query or the main text to process.
            images: A list of local file paths or URLs for images to include.
            json_mode: If True, request a JSON object as the output.

        Returns:
            The text content of the LLM's response.
        """
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        if images:
            for image_ref in images:
                image_url_data: Dict[str, str]
                if image_ref.startswith("http://") or image_ref.startswith("https://"):
                    # It's a URL, pass it directly
                    image_url_data = {"url": image_ref}
                else:
                    # It's a local file path, encode it
                    base64_image = self._encode_image_to_base64(image_ref)
                    image_url_data = {"url": f"data:image/jpeg;base64,{base64_image}"}

                user_content.append({
                    "type": "image_url",
                    "image_url": image_url_data
                })

        messages.append({"role": "user", "content": user_content})

        response_format = {"type": "json_object"} if json_mode else {"type": "text"}

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                response_format=response_format,
            )
        except Exception as e:
            # Fallback for models/gateways that don't support response_format
            if "response_format" in str(e) or "unexpected keyword argument" in str(e):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                )
            else:
                raise e

        return response.choices[0].message.content or ""