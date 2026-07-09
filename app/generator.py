from threading import Thread
from transformers import TextIteratorStreamer

def get_safetensors_input(prompt: str, model_manager):
    tokenizer = model_manager.tokenizer_safetensors
    llm = model_manager.llm_safetensors
    messages = [{"role": "user", "content": prompt}]
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    return tokenizer(text, return_tensors="pt").to(llm.device)

def generate_gguf_stream(prompt: str, max_tokens: int, temperature: float, model_manager):
    llm = model_manager.llm_gguf
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}], 
        max_tokens=max_tokens, 
        temperature=temperature, 
        stream=True
    )
    for chunk in response:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta: 
            yield delta['content']

def generate_gguf_sync(prompt: str, max_tokens: int, temperature: float, model_manager) -> str:
    llm = model_manager.llm_gguf
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}], 
        max_tokens=max_tokens, 
        temperature=temperature, 
        stream=False
    )
    return response['choices'][0]['message']['content']

def generate_safetensors_stream(prompt: str, max_tokens: int, temperature: float, model_manager):
    llm = model_manager.llm_safetensors
    tokenizer = model_manager.tokenizer_safetensors
    inputs = get_safetensors_input(prompt, model_manager)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs, 
        streamer=streamer,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0, 
        pad_token_id=tokenizer.eos_token_id, 
        eos_token_id=tokenizer.eos_token_id
    )
    if temperature > 0: 
        generation_kwargs["temperature"] = temperature
        
    thread = Thread(target=llm.generate, kwargs=generation_kwargs)
    thread.start()
    
    # OPTIMIZATION: Filter out empty strings/whitespace to prevent UI stutter
    for new_text in streamer: 
        if new_text:  # Only yield if there is actual text
            yield new_text
            
    thread.join()

def generate_safetensors_sync(prompt: str, max_tokens: int, temperature: float, model_manager) -> str:
    llm = model_manager.llm_safetensors
    tokenizer = model_manager.tokenizer_safetensors
    inputs = get_safetensors_input(prompt, model_manager)
    generation_kwargs = dict(
        **inputs, 
        max_new_tokens=max_tokens, 
        do_sample=temperature > 0, 
        pad_token_id=tokenizer.eos_token_id, 
        eos_token_id=tokenizer.eos_token_id
    )
    if temperature > 0: 
        generation_kwargs["temperature"] = temperature
    outputs = llm.generate(**generation_kwargs)
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)