from openai import OpenAI
import streamlit as st
from PIL import Image
import io
import base64


client = OpenAI()

st.title("Image to Text")

uploaded_file = st.file_uploader("upload an  image", type = ["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption = "Uploaded Image" ,use_column_width = True)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    img_url = f"data:image/png;base64,{img_str}"

    with st.spinner("Generating description..."):
        response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url,
                    },
                },
                ],
            }],
        )

    description = response.choices[0].message.content
    st.subheader("Description")
    st.write(description)


