# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenAI-Compatible provider for LangExtract."""
# pylint: disable=duplicate-code

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import os
import time
from typing import Any, Iterator, Sequence

import openai
from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions
from langextract.core import schema
from langextract.core import types as core_types
from langextract.providers import router


@router.register(
    r'^oai-compat/',
    priority=10, # Default priority for custom plugins
)
@dataclasses.dataclass(init=False)
class OpenAICompatibleLanguageModel(base_model.BaseLanguageModel):
  """Language model for endpoints compatible with the OpenAI API."""

  model_id: str
  api_key: str | None = None
  base_url: str = "https://oneapi.wanghu.rcfortress.site:8443/v1"
  organization: str | None = None
  format_type: data.FormatType = data.FormatType.JSON
  temperature: float | None = None
  max_workers: int = 10
  _client: Any = dataclasses.field(default=None, repr=False, compare=False)
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  @property
  def requires_fence_output(self) -> bool:
    """OpenAI JSON mode returns raw JSON without fences."""
    if self.format_type == data.FormatType.JSON:
      return False
    return super().requires_fence_output

  def __init__(
      self,
      model_id: str,
      api_key: str | None = None,
      organization: str | None = None,
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float | None = None,
      max_workers: int = 10,
      **kwargs,
  ) -> None:
    """Initialize the OpenAI-compatible language model.

    Args:
      model_id: The model ID to use (e.g., 'moonshotai/kimi-k2-instruct').
      api_key: API key for the service. If not provided, it will try to use
        the ONEAPI_API_KEY environment variable.
      organization: Optional organization ID.
      format_type: Output format (JSON or YAML).
      temperature: Sampling temperature.
      max_workers: Maximum number of parallel API calls.
      **kwargs: Ignored extra parameters.
    """
    # Strip the routing prefix to get the actual model name for the API
    prefix = 'oai-compat/'
    if model_id.startswith(prefix):
      self.model_id = model_id[len(prefix):]
    else:
      self.model_id = model_id
      
    self.api_key = api_key or os.getenv("ONEAPI_API_KEY")
    self.organization = organization
    self.format_type = format_type
    self.temperature = temperature
    self.max_workers = max_workers

    if not self.api_key:
      raise exceptions.InferenceConfigError(
          'API key not provided. Pass it as an argument or set the '
          'ONEAPI_API_KEY environment variable.'
      )

    # Initialize the OpenAI client with the hardcoded base_url
    self._client = openai.OpenAI(
        api_key=self.api_key,
        base_url=self.base_url,
        organization=self.organization,
    )

    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )
    self._extra_kwargs = kwargs or {}

  def _normalize_reasoning_params(self, config: dict) -> dict:
    """Normalize reasoning parameters for API compatibility."""
    result = config.copy()

    if 'reasoning_effort' in result:
      effort = result.pop('reasoning_effort')
      reasoning = result.get('reasoning', {}) or {}
      reasoning.setdefault('effort', effort)
      result['reasoning'] = reasoning

    return result

  def _process_single_prompt(
      self, prompt: str, config: dict
  ) -> core_types.ScoredOutput:
    """Process a single prompt with retries and return a ScoredOutput."""
    max_retries = 3
    last_error = None

    try:
      normalized_config = self._normalize_reasoning_params(config)

      system_message = ''
      if self.format_type == data.FormatType.JSON:
        system_message = (
            'You are a helpful assistant that responds in JSON format.'
        )
      elif self.format_type == data.FormatType.YAML:
        system_message = (
            'You are a helpful assistant that responds in YAML format.'
        )

      messages = [{'role': 'user', 'content': prompt}]
      if system_message:
        messages.insert(0, {'role': 'system', 'content': system_message})

      api_params = {
          'model': self.model_id,
          'messages': messages,
          'n': 1,
      }

      temp = normalized_config.get('temperature', self.temperature)
      if temp is not None:
        api_params['temperature'] = temp

      if self.format_type == data.FormatType.JSON:
        api_params.setdefault('response_format', {'type': 'json_object'})

      if (v := normalized_config.get('max_output_tokens')) is not None:
        api_params['max_tokens'] = v
      if (v := normalized_config.get('top_p')) is not None:
        api_params['top_p'] = v
      for key in [
          'frequency_penalty',
          'presence_penalty',
          'seed',
          'stop',
          'logprobs',
          'top_logprobs',
          'reasoning',
          'response_format',
      ]:
        if (v := normalized_config.get(key)) is not None:
          api_params[key] = v
    except Exception as e:
      # This part should not fail, but as a safeguard
      raise exceptions.InferenceConfigError(f"Error preparing API request: {e}") from e

    for attempt in range(max_retries):
        try:
            response = self._client.chat.completions.create(**api_params)
            output_text = response.choices[0].message.content

            start = output_text.find('{')
            end = output_text.rfind('}')
            if start != -1 and end != -1:
                json_str = output_text[start:end + 1]
                json.loads(json_str) # Validate
                return core_types.ScoredOutput(score=1.0, output=json_str)
            else:
                raise json.JSONDecodeError("No JSON object found in response", output_text, 0)

        except (json.JSONDecodeError, openai.APIError) as e:
            last_error = e
            print(f"Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}. Retrying in 1 second...")
            time.sleep(1)
            continue
            
    # If loop finishes without returning, it means all retries failed.
    raise exceptions.InferenceRuntimeError(
        f'Failed to get a valid JSON response after {max_retries} attempts.', original=last_error
    ) from last_error

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[core_types.ScoredOutput]]:
    """Runs inference on a list of prompts via an OpenAI-compatible API."""
    merged_kwargs = self.merge_kwargs(kwargs)

    config = {}

    temp = merged_kwargs.get('temperature', self.temperature)
    if temp is not None:
      config['temperature'] = temp
    if 'max_output_tokens' in merged_kwargs:
      config['max_output_tokens'] = merged_kwargs['max_output_tokens']
    if 'top_p' in merged_kwargs:
      config['top_p'] = merged_kwargs['top_p']

    for key in [
        'frequency_penalty',
        'presence_penalty',
        'seed',
        'stop',
        'logprobs',
        'top_logprobs',
        'reasoning_effort',
        'reasoning',
        'response_format',
    ]:
      if key in merged_kwargs:
        config[key] = merged_kwargs[key]

    if len(batch_prompts) > 1 and self.max_workers > 1:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=min(self.max_workers, len(batch_prompts))
      ) as executor:
        future_to_index = {
            executor.submit(
                self._process_single_prompt, prompt, config.copy()
            ): i
            for i, prompt in enumerate(batch_prompts)
        }

        results: list[core_types.ScoredOutput | None] = [None] * len(
            batch_prompts
        )
        for future in concurrent.futures.as_completed(future_to_index):
          index = future_to_index[future]
          try:
            results[index] = future.result()
          except Exception as e:
            raise exceptions.InferenceRuntimeError(
                f'Parallel inference error: {str(e)}', original=e
            ) from e

        for result in results:
          if result is None:
            raise exceptions.InferenceRuntimeError(
                'Failed to process one or more prompts'
            )
          yield [result]
    else:
      for prompt in batch_prompts:
        result = self._process_single_prompt(prompt, config.copy())
        yield [result]
