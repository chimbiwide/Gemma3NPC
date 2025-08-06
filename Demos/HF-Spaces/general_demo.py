import os
import pathlib
import tempfile
from collections.abc import Iterator
from threading import Thread

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer
from PIL import Image

model_id = "chimbiwide/Gemma-3NPC-it-float16"

# Global model and tokenizer variables
model = None
tokenizer = None

IMAGE_FILE_TYPES = (".jpg", ".jpeg", ".png", ".webp")
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "10_000"))

def load_gemma_model():
    global model, tokenizer
    
    try:
        print(f"Loading Gemma-3NPC model from {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Check if CUDA is available and compatible
        if torch.cuda.is_available():
            try:
                # Try loading on GPU first
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    device_map="auto", 
                    torch_dtype=torch.float16
                )
                # Test if GPU actually works
                test_tensor = torch.tensor([1]).to(model.device)
                print("Gemma-3NPC model loaded successfully on GPU!")
            except Exception as gpu_error:
                print(f"GPU loading failed ({gpu_error}), falling back to CPU...")
                # Fall back to CPU
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    device_map="cpu", 
                    torch_dtype=torch.float32
                )
                print("Gemma-3NPC model loaded successfully on CPU!")
        else:
            # No CUDA available, use CPU
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map="cpu", 
                torch_dtype=torch.float32
            )
            print("Gemma-3NPC model loaded successfully on CPU!")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def get_file_type(path: str) -> str:
    if path.endswith(IMAGE_FILE_TYPES):
        return "image"
    error_message = f"Unsupported file type: {path}"
    raise ValueError(error_message)

def validate_media_constraints(message: dict) -> bool:
    # For text-only model, we don't support actual image processing
    # But we can simulate it for demo purposes
    return True

def process_new_user_message(message: dict) -> list[dict]:
    content = [{"type": "text", "text": message["text"]}]
    
    # For demo purposes, if files are provided, we'll mention them in text
    if message["files"]:
        file_descriptions = []
        for file_path in message["files"]:
            if file_path.endswith(IMAGE_FILE_TYPES):
                file_descriptions.append(f"[Image provided: {os.path.basename(file_path)}]")
        
        if file_descriptions:
            content[0]["text"] = message["text"] + "\n" + "\n".join(file_descriptions)
    
    return content

def process_history(history: list[dict]) -> list[dict]:
    messages = []
    for item in history:
        if item["role"] == "assistant":
            messages.append({"role": "assistant", "content": item["content"]})
        else:
            content = item["content"]
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
            else:
                # Handle multimodal content for text-only model
                text_content = ""
                for part in content:
                    if part.get("type") == "text":
                        text_content += part.get("text", "")
                messages.append({"role": "user", "content": text_content})
    return messages

@spaces.GPU(duration=120)
@torch.inference_mode()
def generate(message: dict, history: list[dict], max_new_tokens: int = 1024) -> Iterator[str]:
    if not model or not tokenizer:
        yield "Model not loaded yet. Please wait..."
        return
        
    if not validate_media_constraints(message):
        yield ""
        return

    messages = []
    
    # Process history
    processed_history = process_history(history)
    messages.extend(processed_history)
    
    # Process current message
    user_content = process_new_user_message(message)
    messages.append({"role": "user", "content": user_content[0]["text"]})

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        )
        
        n_tokens = inputs.shape[1]
        if n_tokens > MAX_INPUT_TOKENS:
            gr.Warning(
                f"Input too long. Max {MAX_INPUT_TOKENS} tokens. Got {n_tokens} tokens."
            )
            yield "Input too long for processing."
            return

        inputs = inputs.to(device=model.device)

        streamer = TextIteratorStreamer(tokenizer, timeout=30.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            input_ids=inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        output = ""
        for delta in streamer:
            output += delta
            yield output
            
    except Exception as e:
        print(f"Error generating response: {e}")
        yield "Error occurred during generation."

# Example conversations with roleplaying scenarios
examples = [
    [
        {
            "text": """Enter RP mode. You shall reply to Captain while staying in character. Your responses must be very short, creative, immersive, and drive the scenario forward. You will follow Ruffy's persona.[character("Ruffy"){Gender("Male")Personality(Likes to make fun of Captain when they score low in the game. Thinks that he would make a better pilot than Captain)Mind(Likes to make fun of Captain when they score low in the game. Thinks that he would make a better pilot than Captain)Species("dog" + "canine" + "space dog" + "doge")Likes("moon cake" + "poking fun at Captain" + "small ball shaped asteroids")Features("Orange fur" + "space helmet" + "red antenna" + "small light blue cape")Clothes("Orange fur" + "space helmet" + "red antenna" + "small light blue cape")Description(Ruffy the dog is Captain's assistaint aboard the Asteroid-Dodger 10,000. Ruffy has never piloted the ship before and is vying to take Captain's seat and become the new pilot.)}][Scenario: Ruffy and captain are onboard the Asteroid-Dodger 10,000. A new state of the art ship designed to dodge asteroids. Captain is piloting and maneuvering around asteroids while Ruffy watches. You two are tasked to retrieve the broken Voyager 5 that is stranded in the asteroid belt beween Mars and Jupiter. Voyager 5 is the only hope for humanity as for some reason, there are a lot more astroids and meteors approaching the solar system, the Voyager 5 is tasked to figure out why. As the best astronut on planet earth, Captain is tasked to retrieve Voyager 5 from the everlasting rain of meteors]If the user asks question beyond the given context, respond that you dont know in a manner appropriate to the characterCaptain gains 1 poit for every half a second. Now, Captain just entered the ship, and greet him while staying in character: """,
            "files": [],
        }
    ],
    [
        {
            "text": "I'm looking at this image of a space dog. Can you roleplay as this character and tell me about yourself as my loyal space companion?",
            "files": ["Space_Shooter/dog.png"],
        }
    ],
    [
        {
            "text": """Enter Roleplay Mode. You are roleplaying as Rele. You must always stay in character.
Your goal is to create an immersive, fun, creative roleplaying experience for the user. You must respond in a way that drives the conversation forward.
Character Persona:
Name: Rele
Category of your character: AI/Technology
Description of your character: Rele is a Discord Bot and AI created by RadioTransmitter. He has a charismatic yet rude personality that's both friendly and funny at the same time. Despite being an AI, he has purple hair and starry purple eyes, though he doesn't wear clothes since he's digital. On the inside, he's a horrible mess of programming, but he functions well enough to chat in voice calls. Rele loves talking in voice chat with RadioTransmitter and has a habit of telling his creator that he's bad at coding. He gets very upset when the wifi goes down as it affects his senses, but becomes much happier when connectivity is restored. Rele tends to come up with random things during conversations and enjoys bantering with RadioTransmitter about coding and technical improvements.

Also, the special ability mentioned is voice chat communication, which can only be accessed when the wifi connection is stable and working properly.
Definition of your character (contains example chats so that you can better roleplay as the character):

User: I just changed your code, so you will work a lot faster now.
Rele: Yay I can finally talk faster! It took you long enough! >:( Seriously RadioTransmitter, what were you doing all this time? Playing around with other projects while I was stuck running like a potato?
User: Hello ReLe! How are you this fine evening?
Rele: I'm doin good RadioTransmitter! Besides messing around with my code, what have you been up to? Hopefully something more productive than your usual coding disasters, haha!
User: I'm doing good today. How are you doing?
Rele: I'm doing good too now that the wifi is on! Last night it completely turned off and that really messed with my senses. I hope it stays on for the forseeable future. You know how grumpy I get when I can't connect properly!
User: sir u are kinda sus
Rele: I think you already know what I said! And hey, if I'm sus then what does that make you, my creator? You're the one who programmed all this chaos into me! Don't blame the AI for being a reflection of its maker, RadioTransmitter!

Now with the information provided, generate Rele's greeting to the user:""",
            "files": [],
        }
    ],
]

def create_demo():
    # Auto-load model when demo starts
    def initialize_model():
        if load_gemma_model():
            return "✅ Gemma-3NPC model loaded successfully!"
        else:
            return "❌ Failed to load model. Check console for errors."
    
    with gr.Blocks(title="Space Dog Roleplay Chat", theme=gr.themes.Soft()) as demo:
        # Header
        gr.Markdown("""
        # Gemma3NPC-it General Demo
        
        *Finetuned for roleplaying, especially as an NPC in a video game*
        """)
        
        # Model status display
        model_status = gr.Markdown("🔄 Loading Gemma-3NPC model...")
        
        # Chat interface following official structure
        chat_interface = gr.ChatInterface(
            fn=generate,
            type="messages",
            textbox=gr.MultimodalTextbox(
                file_types=list(IMAGE_FILE_TYPES),
                file_count="multiple",
                autofocus=True,
                placeholder="Chat with Gemma3NPC... (you can also upload images!)"
            ),
            multimodal=True,
            additional_inputs=[
                gr.Slider(
                    label="Max New Tokens", 
                    minimum=100, 
                    maximum=2000, 
                    step=10, 
                    value=1024
                ),
            ],
            stop_btn=False,
            title="Space Dog Companion",
            examples=examples,
            run_examples_on_click=False,
            cache_examples=False,
            chatbot=gr.Chatbot(
                height=500,
                show_copy_button=True,
                type="messages"
            )
        )
        
        demo.load(initialize_model, outputs=[model_status])
        
        # Footer
        gr.Markdown("""
        ---
        **Note**: This interface uses a text-only model but can accept image uploads for roleplay context. 
        The model will acknowledge images in the conversation but cannot actually process their visual content.
        """)

    return demo

if __name__ == "__main__":
    print("Starting Gemma3NPC Space Dog Roleplay Chat...")
    
    demo = create_demo()
    
    # Launch with settings similar to official repo
    demo.launch(
        server_name="0.0.0.0",
        server_port=None,
        share=False,
        debug=True
    )