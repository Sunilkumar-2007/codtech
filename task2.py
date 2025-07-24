from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Function to generate text
def generate_text(prompt, max_length=150):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs, 
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# Example usage
prompt = "The future of artificial intelligence in education"
generated_paragraph = generate_text(prompt)
print(generated_paragraph)
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Load pre-trained model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load and preprocess audio
waveform, sample_rate = torchaudio.load("converted.wav")
resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)

# Tokenize input
input_values = tokenizer(waveform.squeeze().numpy(), return_tensors="pt").input_values

# Predict
with torch.no_grad():
    logits = model(input_values).logits

# Decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.decode(predicted_ids[0])

print("Transcribed Text:\n", transcription)
