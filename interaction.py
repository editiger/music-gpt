import sys
import torch
from peft import PeftModel
import transformers
import gradio as gr
import argparse
import warnings
import os


assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
parser.add_argument("--lora_path", type=str, default="./guitarGPT")
parser.add_argument("--use_local", type=int, default=1)
args = parser.parse_args()

tokenizer = LlamaTokenizer.from_pretrained(args.model_path)

LOAD_8BIT = True
BASE_MODEL = args.model_path
LORA_WEIGHTS = args.lora_path

# fix the path for local checkpoint
lora_bin_path = os.path.join(args.lora_path, "adapter_model.bin")
print(lora_bin_path)
if not os.path.exists(lora_bin_path) and args.use_local:
    pytorch_bin_path = os.path.join(args.lora_path, "pytorch_model.bin")
    print(pytorch_bin_path)
    if os.path.exists(pytorch_bin_path):
        os.rename(pytorch_bin_path, lora_bin_path)
        warnings.warn("The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'")
    else:
        assert ('Checkpoint is not Found!')
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        torch_dtype=torch.float16,
        device_map={'': 0}
    )
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )

def generate_prompt(instruction, input=None):
    if input:
        return f"""The following is a conversation between an AI assistant called Assistant and a human user called User.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""The following is a conversation between an AI assistant called Assistant and a human user called User.

### Instruction:
{instruction}

### Response:"""

if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

def interaction(
    input,
    history,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    repetition_penalty=1.0,
    max_memory=256,
    **kwargs,
):
    now_input = input
    history = history or []
    if len(history) != 0:
        input = "\n".join(["User:" + i[0]+"\n"+"Assistant:" + i[1] for i in history]) + "\n" + "User:" + input
        if len(input) > max_memory:
            input = input[-max_memory:]
    print(input)
    print(len(input))
    prompt = generate_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            repetition_penalty=float(repetition_penalty),
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    output = output.split("### Response:")[1].strip()
    output = output.replace("Belle", "Vicuna")
    if 'User:' in output:
        output = output.split("User:")[0]
    history.append((now_input, output))
    print(history)
    return history, history

chatbot = gr.Chatbot().style(color_map=("green", "pink"))
demo = gr.Interface(
    fn=interaction,
    inputs=[
        gr.components.Textbox(
            lines=2, label="Input", placeholder="*64 8H[4] 8K[4] 8F[5] 8H[4] 8F[5] 8H[5] 8F[5] 8K[4] |"
        ),
        "state",
        gr.components.Slider(minimum=0, maximum=1, value=1.0, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
        gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
        gr.components.Slider(minimum=1, maximum=5, step=1, value=2, label="Beams"),
        gr.components.Slider(minimum=1, maximum=2000, step=1, value=128, label="Max new tokens"),
        gr.components.Slider(minimum=0.1, maximum=2.5, value=1.2, label="Repetition Penalty"),
        gr.components.Slider(minimum=0, maximum=256, step=1, value=128, label="max memory"),
    ],
    outputs=[chatbot, "state"],
    allow_flagging="auto",
    title="guitarGPT",
    description="guitarGPT se basa en el modelo [llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf) y se usó la herramienta para ajuste fino [LLaMA-LoRA-Tuner](https://github.com/zetavg/LLaMA-LoRA-Tuner) debido a los pocos requisitos computacionales necesarios. La pretensión del presente proyecto es únicamente hacer una contribución a la comunidad académica.",
)
demo.queue().launch(share=True, inbrowser=True)
