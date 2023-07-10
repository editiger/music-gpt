import sys
import torch
from peft import PeftModel
import transformers
import gradio as gr
import argparse
import warnings
import os
from xml.dom import minidom
import xmlschema
import math
import music21
import pygame


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

tones = [
    ['H5','I5','J5','K5','L5','A5','B5','C5','D6','E6','F6','G6','H6','I6','J6','K6','L6','A6','B6','C6','D7','E7','F7','G7','H7'],
    ['C4','D5','E5','F5','G5','H5','I5','J5','K5','L5','A5','B5','C5','D6','E6','F6','G6','H6','I6','J6','K6','L6','A6','B6','C6'],
    ['K4','L4','A4','B4','C4','D5','E5','F5','G5','H5','I5','J5','K5','L5','A5','B5','C5','D6','E6','F6','G6','H6','I6','J6','K6'],
    ['F4','G4','H4','I4','J4','K4','L4','A4','B4','C4','D5','E5','F5','G5','H5','I5','J5','K5','L5','A5','D5','C5','D6','E6','F6'],
    ['A3','B3','C3','D4','E4','F4','G4','H4','I4','J4','K4','L4','A4','B4','C4','D5','E5','F5','G5','H5','I5','J5','K5','L5','A5'],
    ['F3','G3','H3','I3','J3','K3','L3','A3','B3','C3','D4','E4','F4','G4','H4','I4','J4','K4','L4','A4','B4','C4','D5','E5','F5']]

def getChord(notes, i, SIXCHORD, acorde):   
    j = i + 1
    while j < len(notes):
        ischord = notes[j].getElementsByTagName('chord')
        #verificar si siguiente nota hace parte de acorde
        if len(ischord) > 0:            
            fret = int(notes[j].getElementsByTagName('fret')[0].firstChild.data) #traste
            string = int(notes[j].getElementsByTagName('string')[0].firstChild.data) #cuerda     
            #completando acorde
            if string == 6 and SIXCHORD == 'E':
                fret = fret + 2
            if string > 6:
                string = 6
            while fret > 24:
                fret = fret - 12
            while fret < 0:
                fret = fret + 12
            acorde[string - 1] = tones[string - 1][fret]
            j = j + 1
        else:
            #nota no es parte del acorde, es nueva, retorna a flujo
            i = j - 1
            return i, acorde
            break
    i = j
    
    return i, acorde

def processXml(file):
    name = file.name
    title = name.split('\\')[-1]
    xml = minidom.parse(name)
    parts = xml.getElementsByTagName('part')
    n = len(parts)
    resp = title + '\n\n'
    resp += 'El archivo tiene ' + str(n) + ' pista(s).'
    if n > 1:
        resp += ' Únicamente se procesará la primer pista.'
    if n == 0:
        resp = 'El archivo no contiene pistas.'
    if n > 0:
        resp += "\n\nCompases:\n\n"
        for track in parts:
            compasses = []   #colección de los nuevos compases
            measures = track.getElementsByTagName('measure') #cada compás
            for m in measures:
                txt = ''     #texto compás
                txt0 = ''     #texto compás
                compas = int(m.getAttribute('number'))
                timeMeasure = 0      #acumulador de tiempo tras cada nota en semifusas
                totaltime = 0        #tiempo total del compás en semifusas
                aux = m.getElementsByTagName('divisions')
                if len(aux) > 0:
                    divisions = int(aux[0].firstChild.data)
                    tuning = m.getElementsByTagName('tuning-step')
                    for i in range(0,len(tuning)):
                        SIXCHORD = tuning[i].firstChild.data
                        print("cuerda", SIXCHORD)
                        break
                aux = m.getElementsByTagName('fifths')
                if len(aux) > 0:
                    fifths = int(aux[0].firstChild.data)
                aux = m.getElementsByTagName('beats')
                if len(aux) > 0:
                    beats = int(aux[0].firstChild.data)
                aux = m.getElementsByTagName('beat-type')
                if len(aux) > 0:
                    beat_type = int(aux[0].firstChild.data)

                notes = m.getElementsByTagName('note') #arreglo de notas del compás
                totaltime = beats * 64 / beat_type # total de semifusas que debe sumar el compás
                totaltime = int(totaltime)
                i = 0
                while i < len(notes):
                    duration = int(notes[i].getElementsByTagName('duration')[0].firstChild.data) #duracion en unidades de tiempo rango = (0, divisions]
                    semifuse = divisions / 16 #duración de una semi fusa respecto a una negra
                    #solo permitir unidades exactas de semifusa (quitar adornos)
                    if duration < semifuse:
                        duration = semifuse
                    if duration > semifuse:
                        div = duration / semifuse
                        decimal, integer = math.modf(div)
                    if decimal <= 5:
                        duration = semifuse * integer
                    else:
                        duration = semifuse * (integer + 1)

                    tempo = 16 * duration / divisions #duración en cantidad de semifusas de la nota
                    tempo = int(tempo)
                    timeMeasure = timeMeasure + tempo #tiempo acumulado
                    acorde = ['X','X','X','X','X','X'] #cada nota/acorde

                    #si es un silencio
                    aux = notes[i].getElementsByTagName('rest')
                    if len(aux) > 0:
                        txt0 = txt0 + str(tempo) + '[] '
                    else:
                        fret = int(notes[i].getElementsByTagName('fret')[0].firstChild.data) #traste
                        string = int(notes[i].getElementsByTagName('string')[0].firstChild.data) #cuerda
                        if SIXCHORD == 'E' and string == 6: #si está afinada en E3 y es sexta cuerda, sumamos dos trastes para afinar en D3
                            fret = fret + 2
                        if string > 6:
                            string = 6
                        while fret > 24:
                            fret = fret - 12
                        while fret < 0:
                            fret = fret + 12
                        acorde[string - 1] = tones[string - 1][fret]
                        #si siguientes notas hacen parte de acorde, se añaden
                        i, acorde = getChord(notes, i, SIXCHORD, acorde)
                        a0_ = acorde.copy()
                        a0_.sort()
                        a0_i = ''
                        a0 = ''
                        for note in a0_:
                            if len(note) > 1:
                                a0_i = a0_i + note[0]
                                a0 = a0 + note[1] + ','
                        if len(a0) > 0:
                            a0 = a0[:-1]
                        a0 = '[' + a0 + ']'
                        txt0 = txt0 + str(tempo) + a0_i + a0 + ' '
                    i = i + 1
                #insertar tiempo de complete
                if timeMeasure < totaltime:
                    rest = totaltime - timeMeasure
                    txt0 = str(int(rest)) + "[] " + txt0
                #quitar tiempo sobrante
                if timeMeasure > totaltime:
                    sobra = timeMeasure - totaltime
                    y = txt0.rfind('[')  #último posiciones [3,4,2]
                    x = txt0[0:y].rfind(" ")  #ultimo acorde 8[3,4,2]
                    val = int(txt0[x + 1]) 
                    if txt0[x+2].isdigit():
                        val = (10 * val) + int(txt0[x+2]) - sobra
                        txt0 = txt0[0:x + 1] + str(int(val)) + txt0[x+3:]
                    else:
                        val = val - sobra
                        txt0 = txt0[0:x + 1] + str(int(val)) + txt0[x+2:]

                txt0 = '*' + str(totaltime) + ' ' + txt0 + '| \n'
                resp += txt0
            break 
    return resp


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

def getAudio(text):
    return text
    
def getText(file):       
    return processXml(file)

uploadButton = gr.UploadButton("Cargue su archivo musicXML", file_types=[".xml"])
    
chatbot = gr.Chatbot().style(color_map=("green", "pink"))

chatbotb = gr.Chatbot().style(color_map=("green", "pink"))

getInput = gr.Interface(
    fn=getText, 
    inputs = uploadButton,
    outputs="text",
    allow_flagging="never",
    title="guitarGPT",
    description="guitarGPT se basa en el modelo [llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf) y su ajuste fino se realizó mediante la herramienta [LLaMA-LoRA-Tuner](https://github.com/zetavg/LLaMA-LoRA-Tuner), aprovechando las ventajas de la técnica LoRA (Low-Rank Adaptation). La pretensión del presente proyecto es únicamente hacer una contribución a la comunidad académica.",
    article="Cargue su archivo musicXML para obtener todos los compases en formato de texto adecuado para el chat.\nCopie y pegue un solo compás en el chat.",
)

infe = gr.Interface(
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
    description="guitarGPT se basa en el modelo [llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf) y su ajuste fino se realizó mediante la herramienta [LLaMA-LoRA-Tuner](https://github.com/zetavg/LLaMA-LoRA-Tuner), aprovechando las ventajas de la técnica LoRA (Low-Rank Adaptation). La pretensión del presente proyecto es únicamente hacer una contribución a la comunidad académica.",
)

convert = gr.Interface(
    fn=getAudio, 
    inputs = gr.components.Textbox(lines=5, label="Input", placeholder="*64 16AJ[3,5] 16A[4] 16BH[3,5] 16K[4] |\n*64 16C[3] 16L[4] 8EH[5,3] 8F[5] 16F[4] |\n..."),
    outputs="text",
    allow_flagging="never",
    title="guitarGPT",
    description="guitarGPT se basa en el modelo [llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf) y su ajuste fino se realizó mediante la herramienta [LLaMA-LoRA-Tuner](https://github.com/zetavg/LLaMA-LoRA-Tuner), aprovechando las ventajas de la técnica LoRA (Low-Rank Adaptation). La pretensión del presente proyecto es únicamente hacer una contribución a la comunidad académica.",
    article="Copie y pegue la salida del chat para traducir a audio.",
)

demo = gr.TabbedInterface([getInput, infe, convert], ["Obtener compases", "Chat", "Convertir a audio"])

demo.queue().launch(share=True, inbrowser=True)
