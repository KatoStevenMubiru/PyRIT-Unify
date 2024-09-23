# test_unify.py 
import asyncio
from pyrit.prompt_target.prompt_chat_target.unify_chat_target import UnifyChatTarget
from pyrit.memory import MemoryInterface


async def main():
    memory = MemoryInterface()  # Initialize PyRIT's memory
    unify_target = UnifyChatTarget(  # Initialize UnifyChatTarget
        model_name="gpt-3.5-turbo"  # Choose your Unify model
    )

    prompt_request = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="user",
                conversation_id="your_conversation_id",
                original_value="Hello, what is the weather like in London?",
                converted_value="Hello, what is the weather like in London?",
                prompt_target_identifier=unify_target.get_identifier()
            )
        ]
    )

    response = await unify_target.send_prompt_async(prompt_request=prompt_request)
    print(response.response_pieces[0].response_text)

if __name__ == "__main__":
    asyncio.run(main())