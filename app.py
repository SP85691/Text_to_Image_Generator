import streamlit as st
import shutil
from gradio_client import Client
from PIL import Image

chat_history = []

# Function for generating images from text
class TextToImageGenerator:
    def __init__(self, client, prompt, neg_prompt, stable_diffusion_checkpoints, width, height):
        self.client = client
        self.prompt = prompt
        self.negPrompt = neg_prompt
        self.stable_diffusion_checkpoints = stable_diffusion_checkpoints
        self.sampling_steps = 20.0
        self.sampling_method = "DPM++ 2M Karras"
        self.cfg_scale = 7.0
        self.width = width
        self.height = height
        self.seed = -1.0
        self.result = None

    def generate_image(self):
        self.result = self.client.predict(self.prompt,
                                self.negPrompt,
                                self.stable_diffusion_checkpoints,
                                self.sampling_steps,
                                self.sampling_method,
                                self.cfg_scale,
                                self.width,
                                self.height,
                                self.seed,
                                fn_index = 0
                            )
        return self.result

# Initialize chat history in session state if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

def main():
    st.set_page_config(page_title="Stable Gen AI", page_icon="🤖")
    st.title('Text to Image Generator 🚀✌️')
    st.divider()
    with st.chat_message("assistant"):
        st.write("🤖 Prompt Freely as much you want ✌️")
    model_options = {
        "prodia/sdxl-stable-diffusion-xl": [
            "juggernautXL_v45.safetensors [e75f5471]", 
            "realismEngineSDXL_v10.safetensors [af771c3f]",
            "turbovisionXL_v431.safetensors [78890989]",
            "dynavisionXL_0411.safetensors [c39cc051]",
            "sd_xl_base_1.0.safetensors [be9edd61]",
        ],
        "prodia/fast-stable-diffusion": [
            "amIReal_V41.safetensors [0a8a2e61]",
            "absolutereality_v181.safetensors [3d9d4d2b]",
            "cyberrealistic_v33.safetensors [82b0d085]",
            "anythingV5_PrtRE.safetensors [893e49b9]"
        ],
    }

    selected_model = st.sidebar.selectbox("Choose models", list(model_options.keys()))
    st.sidebar.info(f"Selected Model: {selected_model}")

    model_values = model_options[selected_model]
    selected_checkpoint = st.sidebar.selectbox("Choose a checkpoint", model_values)
    st.sidebar.info("Choose Turbovision for better results 🔥✌️")

    width = st.sidebar.slider("Width", min_value=100, max_value=1000, value=512, step=1)
    height = st.sidebar.slider("Height", min_value=100, max_value=1000, value=512, step=1)

    input_txt = st.chat_input('Enter text to generate image')
    negative_prompt = "ugly, deformed eyes, deformed ears, deformed nose, deformed mouth, bad anatony, extra  limbs, low resolution, pixelated, distorted proportion, unnatural colors, unrealistic lighting, excessive noise, unnatural pose, unrealistic shadows"
    
    if input_txt:
        # Save user prompt to chat history
        st.session_state['chat_history'].append(("User", input_txt))
        client = Client(selected_model)
        app = TextToImageGenerator(client=client, prompt=input_txt, neg_prompt=negative_prompt, stable_diffusion_checkpoints=selected_checkpoint, width=width, height=height)
        result = app.generate_image()
        
        # Save Baymax's response to chat history
        st.session_state['chat_history'].append(("Baymax", result))
        
    for role, Res in st.session_state['chat_history']:
        if role == "User":
            with st.chat_message("user"):
                st.write(Res)
        else:
            message = st.chat_message("assistant")
            try:
                message.write("Here is your generated image")
                image = Image.open(Res)
                message.image(image, caption='Generated Image', use_column_width=True, width=width, clamp=True)
                if message.button("Download Image", key=Res):
                    with open(Res, "rb") as f:
                        bytes = f.read()
                        message.download_button(label="Download", data=bytes, file_name="generated_image.png",
                                            mime="image/png", type="primary")
            except Exception as e:
                message.write(f"Error opening image: {e}")
            
if __name__ == '__main__':
    main()