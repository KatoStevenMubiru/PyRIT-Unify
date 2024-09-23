# Copyright (c) Your Name/Company
# Licensed under the MIT license.

import logging
from typing import Optional, Union, Dict

from pyrit.chat_message_normalizer import ChatMessageNop, ChatMessageNormalizer
from pyrit.common import default_values
from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage, PromptRequestPiece, PromptRequestResponse
from pyrit.models import construct_response_from_request
from pyrit.prompt_target import PromptChatTarget
from unify import Unify  # Import the Unify library

logger = logging.getLogger(__name__)


class UnifyChatTarget(PromptChatTarget):
    """
    Represents a chat target using Unify's API.

    This class enables interaction with Unify's multi-provider LLM platform within the PyRIT framework. 
    It handles sending prompts, receiving responses, and managing conversation history.

    **To use this class:**

    1. **Install Unify:**
       ```bash
       pip install unifyai
       ```
    2. **Set API Key:** 
       * Create an environment variable `UNIFY_API_KEY` and store your Unify API key.
       * You can set this environment variable in your shell (e.g., `.bashrc` or `.zshrc`) or use the 
         `python-dotenv` library to load it from a `.env` file.

    **Example:**

    ```python
    from pyrit.prompt_target import UnifyChatTarget  

    unify_target = UnifyChatTarget(
        api_key="your_unify_api_key",
        model_name="gpt-3.5-turbo"
    )

    prompt_request = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="user", 
                conversation_id="your_conversation_id", 
                original_value="Hello! What's the weather like today?", 
                converted_value="Hello! What's the weather like today?", 
                prompt_target_identifier=unify_target.get_identifier()
            )
        ]
    )

    response = await unify_target.send_prompt_async(prompt_request=prompt_request)
    print(response.response_pieces[0].response_text)
    ```
    """

    API_KEY_ENVIRONMENT_VARIABLE = "UNIFY_API_KEY"

    def __init__(
        self,
        *,
        api_key: str = None,
        chat_message_normalizer: ChatMessageNormalizer = ChatMessageNop(),
        memory: MemoryInterface = None,
        model_name: str = "gpt-3.5-turbo",  # Default Unify model
        **kwargs  # Allow additional Unify parameters
    ) -> None:
        """
        Initializes a Unify chat target.

        Args:
            api_key (str, optional): The Unify API key. Defaults to the
                UNIFY_API_KEY environment variable.
            chat_message_normalizer (ChatMessageNormalizer, optional): The chat message normalizer.
                Defaults to ChatMessageNop().
            memory (MemoryInterface, optional): The memory interface.
                Defaults to None.
            model_name (str, optional): The Unify model name. Defaults to "gpt-3.5-turbo".
            **kwargs: Additional Unify parameters (e.g., temperature, top_p, etc.)
                that can be passed directly to Unify's API.

        Raises:
            ValueError: If the API key is not provided or cannot be found in the environment.
        """
        PromptChatTarget.__init__(self, memory=memory)
        self.api_key = api_key or default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )
        self.chat_message_normalizer = chat_message_normalizer
        self.model_name = model_name
        self.unify_client = Unify(api_key=self.api_key)  # Initialize Unify client
        # Store any additional Unify parameters in kwargs for later use

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Asynchronously sends a prompt to the Unify model and returns the response.

        This method interacts with the Unify API to generate a response based on the provided prompt.
        It retrieves conversation history, formats the prompt for Unify, sends the request, and 
        processes the response.

        Args:
            prompt_request (PromptRequestResponse): The prompt request object, including the conversation
                   history and the current prompt.

        Returns:
            PromptRequestResponse: The response object containing the generated text and other
                   relevant information.
        """
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        messages = self._memory.get_chat_messages_with_conversation_id(
            conversation_id=request.conversation_id
        )
        messages.append(request.to_chat_message())

        logger.info(f"Sending the following prompt to Unify: {request}")

        resp_text = await self.unify_client.generate(
            model=self.model_name,
            messages=messages,
            # Pass any additional Unify parameters from kwargs here
        )
        logger.info(f'Received the following response from Unify: "{resp_text}"')

        return construct_response_from_request(
            request=request, response_text_pieces=[resp_text]
        )

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        """
        Validates the structure of the prompt request.

        Ensures that the prompt request is compatible with Unify's API (e.g., single prompt piece, 
        text data type).

        Args:
            prompt_request (PromptRequestResponse): The prompt request object.
        """
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")