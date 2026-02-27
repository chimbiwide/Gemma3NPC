import gradio as gr
import random
import os

# Try to import required libraries
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    print("Warning: llama-cpp-python not installed.")
    LLAMA_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    print("Warning: huggingface_hub not installed.")
    HF_HUB_AVAILABLE = False

# Global model variable
model = None

def read_scores():
    try:
        with open("scores.txt", "r") as scores:
            score = scores.read()
        return score.strip() if score.strip() else "No scores available yet."
    except FileNotFoundError:
        return "No scores available yet."
    except Exception as e:
        return f"Error reading scores: {str(e)}"

def download_and_find_model():
    model_repo = "chimbiwide/Gemma-3NPC-it-Q4-GGUF"
    
    if not HF_HUB_AVAILABLE:
        print("Error: huggingface_hub not available.")
        return None
    
    try:
        print(f"Downloading/checking model from {model_repo}...")
        
        # Download the entire repository to get all GGUF files
        model_path = snapshot_download(
            repo_id=model_repo,
            repo_type="model",
            allow_patterns=["*.gguf"],  # Only download GGUF files
            local_files_only=False  # Allow downloading if not cached
        )
        
        # Find the GGUF file in the downloaded directory
        for file in os.listdir(model_path):
            if file.endswith('.gguf'):
                full_path = os.path.join(model_path, file)
                print(f"Found GGUF model: {full_path}")
                return full_path
        
        print("No GGUF files found in downloaded model")
        return None
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

def load_gemma_model():
    global model
    
    if not LLAMA_AVAILABLE:
        print("Error: llama-cpp-python not available.")
        return False
    
    # Download/find the model
    model_path = download_and_find_model()
    if not model_path:
        print("Failed to download or find the Gemma-3NPC model")
        return False
    
    try:
        print(f"Loading Gemma-3NPC model from {model_path}...")
        model = Llama(
            model_path=model_path,
            n_ctx=8192,  #context length
            n_gpu_layers=-1,  # Offload all layers to GPU
            n_threads=8,  # Number of CPU threads (for parts not on GPU)
            verbose=False
        )
        print("Gemma-3NPC model loaded successfully on GPU!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def generate_initial_greeting():
    if not model:
        return "Woof! I'm your space dog companion. Loading my AI brain..."
    
    try:
        # System prompt for character definition
        system_prompt = """Enter RP mode. You shall reply to Captain while staying in character. Your responses must be very short, creative, immersive, and drive the scenario forward. You will follow Ruffy's persona.
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
Now Captain just entered the ship, Greet Captain while staying in Character: 
"""
        prompt = f"{system_prompt}\n\nSpace Dog:"
        
        # Generate initial greeting
        response = model(
            prompt,
            max_tokens=150,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            stop=["Pilot:", "\n\n"],
            echo=False
        )
        
        # Extract the generated text
        if response and 'choices' in response and len(response['choices']) > 0:
            greeting = response['choices'][0]['text'].strip()
            return greeting
        else:
            return "Woof! I'm your loyal space dog companion, ready to help you on your missions!"
            
    except Exception as e:
        print(f"Error generating initial greeting: {e}")
        return "Woof! I'm your space dog companion, ready for adventure!"

def chat_with_llm(message, history):
    if not message.strip():
        return history, ""
    
    # Add user message to history
    history = history or []
    history.append({"role": "user", "content": message})
    
    if not model:
        # Fallback response if model not loaded
        response = "Woof! My brain isn't loaded yet. The Gemma-3NPC model might still be downloading..."
        history.append({"role": "assistant", "content": response})
        return history, ""
    
    try:
        # Get current score data
        scores_data = read_scores()
        
        # Build conversation context with updated character prompt
        conversation = """Enter RP mode. You shall reply to Captain while staying in character. Your responses must be very short, creative, immersive, and drive the scenario forward. You will follow Ruffy's persona.
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
Captain gains 1 poit for every half a second.
"""
        
        # Add score information as context
        conversation += f"Pilot's Performance Data:\n{scores_data}\n\n"
        
        # Add recent conversation history (last 10 messages)
        recent_history = history[-10:] if len(history) > 10 else history
        for msg in recent_history[:-1]:  # Exclude the current user message
            if msg["role"] == "user":
                conversation += f"Pilot: {msg['content']}\n"
            else:
                conversation += f"Space Dog: {msg['content']}\n"
        
        # Add current user message
        conversation += f"Pilot: {message}\nSpace Dog:"
        
        # Generate response
        response = model(
            conversation,
            max_tokens=200,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            stop=["Pilot:", "\n\n"],
            echo=False
        )
        
        # Extract the generated text
        if response and 'choices' in response and len(response['choices']) > 0:
            ai_response = response['choices'][0]['text'].strip()
        else:
            ai_response = "Woof! Something went wrong with my circuits. Can you try again?"
        
        # Add AI response to history
        history.append({"role": "assistant", "content": ai_response})
        
    except Exception as e:
        print(f"Error generating response: {e}")
        error_response = "Woof! My AI circuits are having trouble. Technical error occurred."
        history.append({"role": "assistant", "content": error_response})
    
    return history, ""

def create_chat_demo():
    
    with gr.Blocks(
        title="Space Dog Chat",
        theme=gr.themes.Soft()
    ) as demo:

        # Header
        gr.Markdown("""
        # 🐕 Space Dog Companion
        
        *Chat with your AI-powered space dog companion!*
        """)

        # Main chat interface
        # Model status display
        model_status = gr.Markdown("🔄 Loading Gemma-3NPC model...")
        
        chatbot = gr.Chatbot(
            value=[{"role": "assistant", "content": "Woof! I'm your space dog companion. Loading my AI brain..."}],
            height=400,
            show_label=False,
            type="messages"
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Talk to your space companion...",
                show_label=False,
                scale=4
            )
            send_btn = gr.Button("Send", scale=1, variant="primary")
            clear_btn = gr.Button("Clear Chat", scale=1, variant="secondary")

        # Clear chat function
        def clear_chat():
            if model:
                greeting = generate_initial_greeting()
                return [{"role": "assistant", "content": greeting}]
            else:
                return [{"role": "assistant", "content": "Woof! I'm your space dog companion. Loading my AI brain..."}]
        
        # Event handlers
        msg.submit(chat_with_llm, [msg, chatbot], [chatbot, msg])
        send_btn.click(chat_with_llm, [msg, chatbot], [chatbot, msg])
        clear_btn.click(clear_chat, outputs=[chatbot])
        
        # Auto-load model when demo starts
        def initialize_model():
            if load_gemma_model():
                # Generate initial greeting with system prompt
                greeting = generate_initial_greeting()
                return "✅ Gemma-3NPC model loaded successfully!", [{"role": "assistant", "content": greeting}]
            else:
                return "❌ Failed to load model. Check console for errors.", [{"role": "assistant", "content": "Woof! Sorry, I'm having trouble loading my AI brain. Please check the console for technical details."}]
        
        demo.load(initialize_model, outputs=[model_status, chatbot])

    return demo

if __name__ == "__main__":
    print("Starting Gemma3NPC Space Dog Chat...")
    
    # Check if llama-cpp-python is available
    if not LLAMA_AVAILABLE:
        print("\n⚠️  WARNING: llama-cpp-python not installed!")
        print("Install with: pip install llama-cpp-python")
        print("The app will run but won't be able to load models.\n")
    
    demo = create_chat_demo()
    
    # Launch with automatic port selection
    demo.launch(
        server_name="0.0.0.0",
        server_port=None,
        share=True,
        debug=True
    )
