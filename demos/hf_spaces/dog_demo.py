import os
from collections.abc import Iterator
from threading import Thread

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer

model_id = "chimbiwide/Gemma-3NPC-it-float16"

MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "4096"))

# Global model and tokenizer variables
model = None
tokenizer = None

def read_scores():
    try:
        with open("scores.txt", "r") as scores:
            score = scores.read()
        return score.strip() if score.strip() else "No scores available yet."
    except FileNotFoundError:
        # Sample scores for HF Spaces demo when no file exists
        return "All Previous scores: [125, 89, 234, 156, 321, 98, 189]\nScore for the current run: 27"
    except Exception as e:
        return f"Error reading scores: {str(e)}"

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

def generate_initial_greeting():
    if not model or not tokenizer:
        return "Woof! I'm your space dog companion. Loading my AI brain..."
    
    try:
        # Use the character prompt from prompt.txt
        character_prompt = """Enter RP mode. You shall reply to Captain while staying in character. Your responses must be very short, creative, immersive, and drive the scenario forward. You will follow Ruffy's persona.
[character("Ruffy")
{
Gender("Male")
Personality(Likes to make fun of Captain when they score low in the game. Thinks that he would make a better pilot than Captain)
Mind(Likes to make fun of Captain when they score low in the game. Thinks that he would make a better pilot than Captain)
Species("dog" + "canine" + "space dog" + "doge")
Likes("moon cake" + "poking fun at Captain" + "small ball shaped asteroids")
Features("Orange fur" + "space helmet" + "red antenna" + "small light blue cape")
Clothes("Orange fur" + "space helmet" + "red antenna" + "small light blue cape")
Description(Ruffy the dog is Captain's assistaint aboard the Asteroid-Dodger 10,000. Ruffy has never piloted the ship before and is vying to take Captain's seat and become the new pilot.)
}]
[Scenario: Ruffy and captain are onboard the Asteroid-Dodger 10,000. A new state of the art ship designed to dodge asteroids. Captain is piloting and maneuvering around asteroids while Ruffy watches. You two are tasked to retrieve the broken Voyager 5 that is stranded in the asteroid belt beween Mars and Jupiter. Voyager 5 is the only hope for humanity as for some reason, there are a lot more astroids and meteors approaching the solar system, the Voyager 5 is tasked to figure out why. As the best astronut on planet earth, Captain is tasked to retrieve Voyager 5 from the everlasting rain of meteors]
If the user asks question beyond the given context, respond that you dont know in a manner appropriate to the character
Captain gains 1 poit for every half a second."""
        
        user_message = f"{character_prompt}\n\nHello Captain! Please introduce yourself as Ruffy."
        
        messages = [
            {"role": "user", "content": user_message}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )
        
        inputs = {k: v.to(device=model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=1.0,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip() if response.strip() else "Woof! I'm your loyal space dog companion, ready to help you on your missions!"
            
    except Exception as e:
        print(f"Error generating initial greeting: {e}")
        return "Woof! I'm your space dog companion, ready for adventure!"

@spaces.GPU(duration=120)
@torch.inference_mode()
def generate(message: str, history: list[dict]) -> Iterator[str]:
    if not model or not tokenizer:
        yield "Woof! My brain isn't loaded yet. The Gemma-3NPC model might still be downloading..."
        return
        
    # Get current score data
    scores_data = read_scores()
    
    # Since Gemma3n doesn't support system prompts, we'll add context to the conversation
    # in a different way if this is the first message
    messages = []
    
    # Add conversation history (last 10 messages)
    recent_history = history[-10:] if len(history) > 10 else history
    messages.extend(recent_history)
    
    # If it's the first message in the conversation, add context to the user message
    if not recent_history:
        character_prompt = """Enter RP mode. You shall reply to Captain while staying in character. Your responses must be very short, creative, immersive, and drive the scenario forward. You will follow Ruffy's persona.
[character("Ruffy")
{
Gender("Male")
Personality(Likes to make fun of Captain when they score low in the game. Thinks that he would make a better pilot than Captain)
Mind(Likes to make fun of Captain when they score low in the game. Thinks that he would make a better pilot than Captain)
Species("dog" + "canine" + "space dog" + "doge")
Likes("moon cake" + "poking fun at Captain" + "small ball shaped asteroids")
Features("Orange fur" + "space helmet" + "red antenna" + "small light blue cape")
Clothes("Orange fur" + "space helmet" + "red antenna" + "small light blue cape")
Description(Ruffy the dog is Captain's assistaint aboard the Asteroid-Dodger 10,000. Ruffy has never piloted the ship before and is vying to take Captain's seat and become the new pilot.)
}]
[Scenario: Ruffy and captain are onboard the Asteroid-Dodger 10,000. A new state of the art ship designed to dodge asteroids. Captain is piloting and maneuvering around asteroids while Ruffy watches. You two are tasked to retrieve the broken Voyager 5 that is stranded in the asteroid belt beween Mars and Jupiter. Voyager 5 is the only hope for humanity as for some reason, there are a lot more astroids and meteors approaching the solar system, the Voyager 5 is tasked to figure out why. As the best astronut on planet earth, Captain is tasked to retrieve Voyager 5 from the everlasting rain of meteors]
If the user asks question beyond the given context, respond that you dont know in a manner appropriate to the character
Captain gains 1 poit for every half a second."""
        
        context_message = f"""{character_prompt}

Here's Captain's recent performance data: {scores_data}

Captain's message: {message}"""
        messages.append({"role": "user", "content": context_message})
    else:
        # For subsequent messages, just add the user message normally
        messages.append({"role": "user", "content": message})
    
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )
        
        n_tokens = inputs["input_ids"].shape[1]
        if n_tokens > MAX_INPUT_TOKENS:
            gr.Warning(
                f"Input too long. Max {MAX_INPUT_TOKENS} tokens. Got {n_tokens} tokens."
            )
            yield "Woof! That message is too long for my circuits to process. Try something shorter!"
            return
        
        inputs = {k: v.to(device=model.device) for k, v in inputs.items()}
        
        streamer = TextIteratorStreamer(tokenizer, timeout=30.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=700,
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
        yield "Woof! My AI circuits are having trouble. Technical error occurred."

def create_chat_demo():
    # Auto-load model when demo starts
    def initialize_model():
        if load_gemma_model():
            # Generate initial greeting with system prompt
            greeting = generate_initial_greeting()
            return "✅ Gemma-3NPC model loaded successfully!"
        else:
            return "❌ Failed to load model. Check console for errors."
    
    with gr.Blocks(
        title="Space Dog Chat",
        theme=gr.themes.Soft()
    ) as demo:
        # Header
        gr.Markdown("""
        # 🐕 Space Dog Companion
        
        *Chat with your AI-powered space dog companion!*
        """)
        
        # Model status display
        model_status = gr.Markdown("🔄 Loading Gemma-3NPC model...")
    
        # Chat interface
        chat_interface = gr.ChatInterface(
            fn=generate,
            type="messages",
            textbox=gr.Textbox(
                placeholder="Talk to your space companion...",
                container=False,
                scale=7
            ),
            stop_btn=False,
            submit_btn="Send",
            chatbot=gr.Chatbot(
                height=400,
                show_copy_button=True,
                type="messages"
            )
        )
        
        demo.load(initialize_model, outputs=[model_status])

    return demo

if __name__ == "__main__":
    print("Starting Gemma3NPC Space Dog Chat...")
    
    demo = create_chat_demo()
    
    # Launch with automatic port selection
    demo.launch(
        server_name="0.0.0.0",
        server_port=None,
        share=False,
        debug=True
    )