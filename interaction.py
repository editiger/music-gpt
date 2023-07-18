import sys
import torch
from peft import PeftModel
import transformers
import gradio as gr
import argparse
import warnings
import os
from xml.dom import minidom
import math
import music21
from IPython.display import Audio
from collections import defaultdict
from mido import MidiFile
from pydub import AudioSegment
from pydub.generators import Sine
import shutil

from utils.callbacks import Iteratorize, Stream

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
parser.add_argument("--lora_path", type=str, default="editigerun/guitarGPT")
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
        return f"""La siguiente es la conversación entre guitarGPT y un humano.

### instruction:
{instruction}

### completion:"""

if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

duration = {
    1:['1','64th'],2:['2','32nd'],3:['2.','32nd'],4:['4','16th'],5:['4+1',''],6:['4.','16th'],7:['4..','16th'],
    8:['8','eighth'],9:['8+1',''],10:['8+2',''],11:['8+2.',''],12:['8.','eighth'],13:['8.+1',''],14:['8..','eighth'],
    15:['8…','eighth'],16:['16','quarter'],17:['16+1',''],18:['16+2',''],19:['16+2.',''],20:['16+4',''],21:['8..+4..',''],
    22:['8..+8',''],23:['16+4..',''],24:['16.','quarter'],25:['16.+1',''],26:['16.+2',''],27:['16+2.',''],
    28:['16..','quarter'],29:['16..+1',''],30:['16…','quarter'],31:['16….','quarter'],32:['32','half'],33:['32+1','']
    ,34:['32+2',''],35:['32+2.',''],36:['32+4',''],37:['16….+4.',''],38:['32+4.',''],39:['32+4..',''],40:['32+8',''],
    41:['32+8+1',''],42:['16…+8.',''],43:['16….+8.',''],44:['32+8.',''],45:['16….+8..',''],46:['32+8..',''],
    47:['32+8…',''],48:['32.','half'],49:['32.+1',''],50:['32.+2',''],51:['32.+2.',''],52:['32.+4',''],53:['32.+4+1',''],
    54:['32.+4.',''],55:['32.+4..',''],56:['32..','half'],57:['32..+1',''],58:['32..+2',''],59:['32..+2.',''],
    60:['32…','half'],61:['32…+1',''],62:['32….','half'],63:['32…..','half'],64:['64','whole'],65:['64+1',''],
    66:['64+2',''],67:['64+2.',''],68:['64+4',''],69:['64+4+1',''],70:['64+4.',''],71:['64+4..',''],72:['64+8',''],
    73:['64+8+1',''],74:['64+8+2',''],75:['64+8+2.',''],76:['64+8.',''],77:['64+8.+1',''],78:['64+8..',''],79:['64+8…',''],
    80:['64+16',''],81:['64+16+1',''],82:['64+16+2',''],83:['64+16+2.',''],84:['64+16+4',''],85:['64+8..+4..',''],
    86:['64+8..+8',''],87:['64+16+4..',''],88:['64+16.',''],89:['64+16.+1',''],90:['64+16.+2',''],91:['64+16+2.',''],
    92:['64+16..',''],93:['64+16..+1',''],94:['64+16…',''],95:['64+16….',''],96:['64.','whole'],97:['64.+1',''],98:['64.+2',''],
    99:['64.+2.',''],100:['64.+4',''],101:['64.+4+1',''],102:['64.+4.',''],103:['64.+4..',''],104:['64.+8',''],
    105:['64.+8+1',''],106:['64.+8+2',''],107:['64.+8+2.',''],108:['64.+8.',''],109:['64.+8.+1',''],110:['64.+8..',''],
    111:['64.+8…',''],112:['64..','whole'],113:['64..+1',''],114:['64..+2',''],115:['64..+2.',''],116:['64..+4',''],
    117:['64..+4+1',''],118:['64..+4.',''],119:['64..+4..',''],120:['64..+8','']}

#25 trastes
tones = [
    ['E5','F5','F5#','G5','G5#','A5','A5#','B5','C6','C6#','D6','D6#','E6','F6','F6#','G6','G6#','A6','A6#','B6','C7','C7#','D7','D7#','E7'],
    ['B4','C5','C5#','D5','D5#','E5','F5','F5#','G5','G5#','A5','A5#','B5','C6','C6#','D6','D6#','E6','F6','F6#','G6','G6#','A6','A6#','B6'],
    ['G4','G4#','A4','A4#','B4','C5','C5#','D5','D5#','E5','F5','F5#','G5','G5#','A5','A5#','B5','C6','C6#','D6','D6#','E6','F6','F6#','G6'],
    ['D4','D4#','E4','F4','F4#','G4','G4#','A4','A4#','B4','C5','C5#','D5','D5#','E5','F5','F5#','G5','G5#','A5','A5#','B5','C6','C6#','D6'],
    ['A3','A3#','B3','C4','C4#','D4','D4#','E4','F4','F4#','G4','G4#','A4','A4#','B4','C5','C5#','D5','D5#','E5','F5','F5#','G5','G5#','A5'],
    ['D3','D3#','E3','F3','F3#','G3','G3#','A3','A3#','B3','C4','C4#','D4','D4#','E4','F4','F4#','G4','G4#','A4','A4#','B4','C5','C5#','D5']]

tinesGuitarGPT = [
    ['H5','I5','J5','K5','L5','A5','B5','C5','D6','E6','F6','G6','H6','I6','J6','K6','L6','A6','B6','C6','D7','E7','F7','G7','H7'],
    ['C4','D5','E5','F5','G5','H5','I5','J5','K5','L5','A5','B5','C5','D6','E6','F6','G6','H6','I6','J6','K6','L6','A6','B6','C6'],
    ['K4','L4','A4','B4','C4','D5','E5','F5','G5','H5','I5','J5','K5','L5','A5','B5','C5','D6','E6','F6','G6','H6','I6','J6','K6'],
    ['F4','G4','H4','I4','J4','K4','L4','A4','B4','C4','D5','E5','F5','G5','H5','I5','J5','K5','L5','A5','D5','C5','D6','E6','F6'],
    ['A3','B3','C3','D4','E4','F4','G4','H4','I4','J4','K4','L4','A4','B4','C4','D5','E5','F5','G5','H5','I5','J5','K5','L5','A5'],
    ['F3','G3','H3','I3','J3','K3','L3','A3','B3','C3','D4','E4','F4','G4','H4','I4','J4','K4','L4','A4','B4','C4','D5','E5','F5']]


fraction=[[4,'1/16'], [6,'3/32'], [8,'1/8'], [16,'1/4'],[24,'3/8'],[32,'2/4'],[40,'3/2'],[48,'3/4'],[56,'5/2'],[64,'4/4'],[72,'7/2'],[80,'5/4'],[88,'9/2'],[96,'6/4'],[104,'11/2'],[112,'7/4'],[128,'8/4'],[144,'9/4'],[160,'10/4'],[176,'11/4'],[192,'12/4']
]

tempo = 120 # bpm
mid = None
output = None

def note_to_freq(note, concert_A=440.0):
    return (2.0 ** ((note - 69) / 12.0)) * concert_A

def ticks_to_ms(ticks):
    global tempo
    global mid
    tick_ms = (60000.0 / tempo) / mid.ticks_per_beat
    return ticks * tick_ms

def midiToWav():
    global mid
    global output
    output = AudioSegment.silent(mid.length * 1000.0)
    try:
        for track in mid.tracks:
            current_pos = 0.0

            current_notes = defaultdict(dict)

            for msg in track:
                current_pos += ticks_to_ms(msg.time)

                if msg.type == 'note_on':
                    current_notes[msg.channel][msg.note] = (current_pos, msg)

                if msg.type == 'note_off':
                    start_pos, start_msg = current_notes[msg.channel].pop(msg.note)

                    duration = current_pos - start_pos

                    signal_generator = Sine(note_to_freq(msg.note))
                    rendered = signal_generator.to_audio_segment(duration=duration-50, volume=-20).fade_out(100).fade_in(30)

                    output = output.overlay(rendered, start_pos)
    except:
        pass

def setSilence(time, xml):
    txt = duration[time] #["8..", "eighth"]
    if txt[1] != '':   #tiene símbolo es una única figura
        dot = txt[0].split('.')
        xml = xml + '<note><rest/><voice>1</voice>'
        dot = txt[0].split('.')
        xml = xml + '<duration>'+dot[0]+'</duration>'
        xml = xml + '<type>'+txt[1]+'</type>'
        if len(dot) > 1:
            for i in range(1, len(dot)):
                xml = xml + '<dot/>'
        xml = xml + '</note>'
        return xml
    #["32.+4+1", ""], half. + 16th + 64th
    
    txt = txt[0].split('+')
    for i in range(0, len(txt)):
        dot = txt[i].split('.')
        xml = xml + '<note><rest/><voice>1</voice>'
        xml = xml + '<duration>'+dot[0]+'</duration>'
        for k,v in duration.items():
            if v[0] == dot[0]:
                xml = xml + '<type>'+v[1]+'</type>' 
                break
        if len(dot) > 1:
            for i in range(1, len(dot)):
                xml = xml + '<dot/>'
        xml = xml + '</note>'
    return xml

#setNote(time, xmlNotes, measure[ind][1], measure[ind][2]) duración nota, letras, octavas
def setNote(time, xml, letras, posiciones):
    #  [16, 'DK', ['4', '3']]
    notas = []
    for i in range(len(letras)):
        notas.append(letras[i] + posiciones[i])
    txt = duration[time] #['1','64th']
    chords = []
    if txt[1] != '':   #tiene símbolo es una única figura
        cont = 0
        for n in notas:
            fret = -1
            for i in range(0,6):         #cada cuerda
                if i not in chords:   #cuerda ya usada
                    try:
                        fret = tinesGuitarGPT[i].index(n)
                    except ValueError:
                        fret = -1
                    if fret != -1:
                        chords.append(i)
                        note = tones[i][fret]    #A3
                        dot = txt[0].split('.')
                        xml = xml + '<note><pitch><step>'+note[0]+'</step><octave>'+note[1]+'</octave>'
                        if len(note) == 3:  #sostenido
                            xml = xml + '<alter>1</alter>'
                        xml = xml + '</pitch>'
                        xml = xml + '<notations><technical><fret>'+str(fret)+'</fret><string>'+str(i+1)+'</string></technical></notations>'
                        xml = xml + '<voice>1</voice><duration>'+str(time)+'</duration><type>'+txt[1]+'</type>'

                        if len(dot) > 1:
                            for i in range(1, len(dot)):
                                xml = xml + '<dot/>'
                        if cont > 0:
                            xml = xml + '<chord/>'
                        cont = cont + 1
                        xml = xml + '</note>'
                        break
        return xml
    else:
        #["32.+4+1", ""], half. + 16th + 64th 21H5
        txt2 = txt[0].split('+') #[8.., 4..]
        contDot = 0
        for k in range(0, len(txt2)):
            cont = 0
            for n in notas: #[H5,I3]
                fret = -1
                for i in range(0,6):         #cada cuerda
                    if i not in chords:   #cuerda ya usada
                        try:
                            fret = tinesGuitarGPT[i].index(n)
                        except ValueError:
                            fret = -1
                        if fret != -1:
                            chords.append(i)
                            note = tones[i][fret]    #A3
                            dot = txt2[contDot].split('.')
                            xml = xml + '<note><pitch><step>'+note[0]+'</step><octave>'+note[1]+'</octave>'
                            if len(note) == 3:  #sostenido
                                xml = xml + '<alter>1</alter>'
                            xml = xml + '</pitch>'
                            xml = xml + '<notations><technical><fret>'+str(fret)+'</fret><string>'+str(i+1)+'</string></technical></notations>'
                            xml = xml + '<voice>1</voice><duration>'+dot[0]+'</duration>'
                            for k,v in duration.items():
                                if v[0] == dot[0]:
                                    xml = xml + '<type>'+v[1]+'</type>' 

                            if len(dot) > 1:
                                for i in range(1, len(dot)):
                                    xml = xml + '<dot/>'
                            if cont > 0:
                                xml = xml + '<chord/>'
                            xml = xml + '</note>'
                            cont = cont + 1
            contDot = contDot + 1
        return xml
    return xml


def getCompasesOk(txt):
    compases = txt.split('\n')
    measures = []
    resp = 'El texto ha sido correctamente procesado.\n\nDe clic en el botón play para reproducir la obra generada.'
    errores = ''
    cont = 0
    flag_error = False
    for compas in compases:
        compas = compas.strip() #quitar espacios en blanco
        notes = compas.split()
        tempos = []
        measure = []
        measure.append(0)
        cont = cont + 1
        for n in notes:
            tempos.append(n)
        if len(compas) > 0 and len(tempos) > 0 and len(tempos[0]) > 0 and tempos[0][0] == '*' and tempos[-1] == '|': #inicio y fin de compás
            total = -1
            suma = 0
            error = ''
            tempos.pop()
            for note in tempos:
                if total == -1:
                    totalf = tempos[0].replace("*", "") #duración del compás
                    if totalf.isdigit():
                        total = int(totalf)
                    measure[0] = total
                else:
                    #24HI[4,1]
                    i_0 = note.find("[")
                    i_f = note.find("]")
                    if i_f + 1  == len(note) and i_0 != -1: #array posiciones u octavas de la nota
                        k = ''
                        for h in note:
                            if h.isdigit():
                                k = k + h
                            else:
                                break
                        letras = note[len(k):i_0]
                        octavas_ = note[i_0 + 1:-1]
                        if octavas_.find(",") != -1:
                            octa = octavas_.split(',')
                        else:
                            octa = octavas_
                        octavas = []
                        for oc in octa:
                            octavas.append(oc)
                        if len(letras) == len(octavas): #nota bien formada
                            flag = True
                            le = ''
                            for l in letras:
                                permitidas = 'ABCDEFGHIJKL'
                                if permitidas.find(l) == -1:
                                    flag = False
                                    le = l
                                    break
                            if flag:
                                suma += int(k)   #duracion acumulada notas
                                measure.append([int(k), letras, octavas])
                            else:
                                error = 'Nota no existente en sistema de guitarGPT (' + le + ')'
                        else:
                            error = 'Cantidad de notas no es igual a array de posiciones (' + letras + '[' + octavas_ + '])'
                    else:
                        error = 'Compás mal formado'
            if measure[0] == suma:
                measures.append(measure)
            else:
                flag_error = True
                errores = errores + 'Compás ' + str(cont) + ' "' + compas + '": La duración indicada del compás (' + str(measure[0])  + ') no es igual a la suma de la duración de las notas/acordes del mismo (' + str(suma) + ').\n'
        else:
            flag_error = True
            errores = errores + 'Compás ' + str(cont) + ' "' + compas + '": Compás vacío o con estructura errada. \n'
    
    if not flag_error:
        errores = resp
    else:
        errores = 'El texto ha sido procesado omitiendo los siguientes compases (frases): \n\n' + errores
    return measures, errores

def getXml(measures):
    global mid
    global output
    
    xml = '<?xml version="1.0" encoding="UTF-8" standalone="no"?><score-partwise><work><work-title/></work><identification><encoding><software>guitarGPT</software></encoding><creator type="composer"/></identification><part-list>'
    xml = xml + '<score-part id="P1"><part-name/><score-instrument id="P1-I1"><instrument-name>#1</instrument-name></score-instrument><midi-instrument id="P1-I1"><midi-channel>1</midi-channel><midi-program>25</midi-program></midi-instrument></score-part></part-list>'
    xml = xml + '<part id="P1">'
    
    contMeasure = 1
    num = '0'         #numerador beats
    den = '0'         #denominador beat-type
    
    header = '<attributes><divisions>16</divisions><key><fifths>0</fifths><mode>major</mode></key><clef><sign>G</sign><line>2</line></clef><time>'
    footer = '</time><staff-details><staff-lines>6</staff-lines>'
    footer = footer + '<staff-tuning line="1"><tuning-step>D</tuning-step><tuning-octave>3</tuning-octave></staff-tuning>'
    footer = footer + '<staff-tuning line="2"><tuning-step>A</tuning-step><tuning-octave>3</tuning-octave></staff-tuning>'
    footer = footer + '<staff-tuning line="3"><tuning-step>D</tuning-step><tuning-octave>4</tuning-octave></staff-tuning>'
    footer = footer + '<staff-tuning line="4"><tuning-step>G</tuning-step><tuning-octave>4</tuning-octave></staff-tuning>'
    footer = footer + '<staff-tuning line="5"><tuning-step>B</tuning-step><tuning-octave>4</tuning-octave></staff-tuning>'
    footer = footer + '<staff-tuning line="6"><tuning-step>E</tuning-step><tuning-octave>5</tuning-octave></staff-tuning>'
    footer = footer + '</staff-details></attributes><direction placement="above"><sound tempo="120"/></direction>'
    
    for measure in measures:
        xml = xml + '<measure number="'+str(contMeasure)+'">'
        timeMeasure = measure[0]  #tiempo total del compás
        xmlNotes = ''
        for ind in range(1, len(measure)):
            #[16, 'D', ['5']], [16, 'DK', ['4', '3']], [16, '', []]
            time = measure[ind][0]    #duración de la nota/acorde
            if measure[ind][2] == []:
                xmlNotes = setSilence(time, xmlNotes)
            else:
                xmlNotes = setNote(time, xmlNotes, measure[ind][1], measure[ind][2])
        changeFrac = 0
        for k in range(0, len(fraction)):
            val = fraction[k]
            nums = val[1].split('/')
            if val[0] == timeMeasure:
                if num != nums[0] or den != nums[1]:
                    num = nums[0]
                    den = nums[1]
                    changeFrac = 1
                    break
        if changeFrac == 1:
            if contMeasure == 1:
                xml = xml + header + '<beats>'+num+'</beats><beat-type>'+den+'</beat-type>'
                xml = xml + footer
            else:
                xml = xml + '<attributes><time><beats>'+num+'</beats><beat-type>'+den+'</beat-type></time></attributes>'
        xml = xml + xmlNotes + '</measure>'
        contMeasure = contMeasure + 1
    xml = xml + '</part>'
    xml = xml + '</score-partwise>'
    xmlFile = open("/content/guitarGPT/utils/guitarGPT.xml", "w")
    xmlFile.write(xml)
    xmlFile.close()
    try:
        c = music21.converter.parse("/content/guitarGPT/utils/guitarGPT.xml")
        c.write('midi', "/content/guitarGPT/utils/guitarGPT.mid")
        mid = MidiFile("/content/guitarGPT/utils/guitarGPT.mid")
        midiToWav()
        output.export("/content/guitarGPT/utils/guitarGPT.wav", format="wav")
    except OSError as err:
        print("OS error:", err)
    except ValueError:
        print("Could not convert data to an integer.")
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    
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
            acorde[string - 1] = tinesGuitarGPT[string - 1][fret]
            j = j + 1
        else:
            #nota no es parte del acorde, es nueva, retorna a flujo
            i = j - 1
            return i, acorde
            break
    i = j
    
    return i, acorde

def processXml(file):
    global output
    global mid
    name = file.name
    title = name.split('/')[-1]
    resp = ''
    try:
        c = music21.converter.parse(name)
        c.write('midi', "/content/guitarGPT/utils/entra.mid")
    except OSError as err:
        print("OS error:", err)
        resp += "OS error: " + err + '\n\n'
    except ValueError:
        print("Could not convert data to an integer.")
        resp += "Could not convert data to an integer.\n\n"
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        resp += "Unexpected err= " + err + " type " + type(err) + ".\n\n"
    xml = minidom.parse(name)
    parts = xml.getElementsByTagName('part')
    strings = xml.getElementsByTagName('string')
    if len(strings) == 0:
        errores = "Archivo xml no válido. Debe ingresar un archivo musicxml para guitarra.\n"
        return errores, "/content/guitarGPT/utils/entra.mid", "/content/guitarGPT/utils/entra.wav"
    n = len(parts)
    resp += title + '\n\n'
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
                    semifuse = int(semifuse)
                    #1 1.0
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
                        acorde[string - 1] = tinesGuitarGPT[string - 1][fret]
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
                    txt0 = txt0 + str(int(rest)) + "[] "
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
    
    mid = MidiFile("/content/guitarGPT/utils/entra.mid")
    midiToWav()
    output.export("/content/guitarGPT/utils/entra.wav", format="wav")
    return resp, "/content/guitarGPT/utils/entra.mid", "/content/guitarGPT/utils/entra.wav"

def getAudio(txt):
    measures, errores = getCompasesOk(txt)
    if len(measures) > 0:
        getXml(measures)
         
    return errores, "/content/guitarGPT/utils/guitarGPT.mid", "/content/guitarGPT/utils/guitarGPT.xml", "/content/guitarGPT/utils/guitarGPT.wav"


#btn.click(interaction, inputs=[inp, temp, topp, topk, beams, tokens, repet, stream], outputs=chatbot)
def interaction(
    input,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=500,
    repetition_penalty=1.0,
    stream=False,
    **kwargs,
):
    torch.cuda.empty_cache()

    history = ''
    now_input = input.replace("\n", " ")
    history = history or []

    prompt = generate_prompt(now_input)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty":float(repetition_penalty),
      }

    if stream:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
        # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

        def generate_with_callback(callback=None, **kwargs):
            kwargs.setdefault(
                "stopping_criteria", transformers.StoppingCriteriaList()
            )
            kwargs["stopping_criteria"].append(
                Stream(callback_func=callback)
            )
            with torch.no_grad():
                model.generate(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(
                generate_with_callback, kwargs, callback=None
            )

        with generate_with_streaming(**generate_params) as generator:
            for output in generator:
                # new_tokens = len(output) - len(input_ids[0])
                decoded_output = tokenizer.decode(output)

                if output[-1] in [tokenizer.eos_token_id]:
                    break
                decoded_output = decoded_output.replace("| ", "|\n")
                yield decoded_output          
        return  # early return for stream_output

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
   
    now_input = now_input.replace("| ", "|\n")
    output = output.replace("| ", "|\n")
    history.append((now_input, output))

    return history

with gr.Blocks() as demo:
    gr.HTML("<div align='center'><bold><h1>guitarGPT</h1></bold></div>")
    gr.Markdown("guitarGPT se basa en el modelo [llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf) y su ajuste fino se realizó mediante la herramienta [LLaMA-LoRA-Tuner](https://github.com/zetavg/LLaMA-LoRA-Tuner), aprovechando las ventajas de la técnica LoRA (Low-Rank Adaptation).")
    gr.HTML("<p>La pretensión del presente proyecto es únicamente hacer una contribución a la comunidad académica.</p>")
    
    with gr.Tab("Get text"):
                
        def getText(file):
            return processXml(file) 
        
        gr.HTML("<div><p>Cargue su archivo musicXML para guitarra y se generará el correspondiente texto para poder interactuar con guitarGPT.</p><br><p>Una vez generado el texto solo use un compás (una de las líneas) para ingresarla en el chat de guitarGPT</p></div>")
        
        with gr.Row():
            with gr.Column(scale=10):
                out = gr.Textbox()
                with gr.Row():
                    with gr.Column():            
                        midiInput = gr.File("/content/guitarGPT/utils/entra.mid", label="midi")
                    with gr.Column():
                        playInput = gr.Audio("/content/guitarGPT/utils/entra.wav")
                with gr.Row():
                    with gr.Column():
                        midi = gr.Video("/content/guitarGPT/utils/tuxGuitar.avi", label="Obtener archivo musicXML con Tux Guitar")
                    with gr.Column():
                        xml = gr.Video("/content/guitarGPT/utils/funcionamiento.avi", label="Proceso general")
                with gr.Row():
                    f1 = gr.File("/content/guitarGPT/utils/mozartmenuett.xml", label="Ejemplo xml 1")
                    f2 = gr.File("/content/guitarGPT/utils/aguadoop6leccionno30.xml", label="Ejemplo xml 2")

            with gr.Column(scale=1):
                upload_button = gr.UploadButton("Click to Upload a File", file_types=[".xml"], file_count="single")
                upload_button.upload(getText, upload_button, [out, midiInput, playInput])
        
    
    with gr.Tab("Chat"):
        
        gr.HTML("<div><p>Pegue solo una la de las líneas (compás) generadas en la pestaña anterior</p><br><p>El texto generado cópielo y péguelo en la siguiente pestaña para obtener el audio correspondiente.</p></div>")
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Textbox(interactive=True, lines=15, label="guitarGPT", placeholder="*64 8H[4] 8K[4] 8F[5] 8H[4] 8F[5] 8H[5] 8F[5] 8K[4] |\n*64 32D[4] 32[] | \n*64 4A[3] 4A[4] 4D[5] 4A[4] 4H[5] 4D[5] 4A[4] 4D[5] 4H[5] 4D[5] 4A[4] 4D[5] 4H[5] 4D[5] 4A[4] 4D[5] | ")
            
            with gr.Column(scale=1):
                inp = gr.Textbox(lines=2, label="Input", placeholder="*64 8H[4] 8K[4] 8F[5] 8H[4] 8F[5] 8H[5] 8F[5] 8K[4] |")
                temp = gr.Slider(minimum=0, maximum=2, step=0.01, value=1.0, label="Temperature", info="Controla la aleatoriedad: los valores más altos hace que el modelo genere salidas más diversas y aleatorias. A medida que la temperatura se acerca a cero, el modelo se volverá más determinista y repetitivo.")
                topp = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.75,  label="Top p", info="Controla la diversidad a través del muestreo del núcleo: solo se consideran los tokens cuya probabilidad acumulada excede a 'top_p'. 0.5 significa que se consideran la mitad de todas las opciones ponderadas por probabilidad. Solo surtirá efecto si la temperatura se establece en > 0")
                topk = gr.Slider(minimum=0, maximum=100, step=0.01, value=40, label="Top k", info="Controla la diversidad de texto generado considerando solo los 'top_k' tokens con las probabilidades más altas. Este método puede generar resultados más centrados y coherentes al reducir el impacto de los tokens de baja probabilidad. Solo tendrá efecto si la temperatura se establece en > 0.")
                beams = gr.Slider(minimum=1, maximum=5, step=1, value=2, label="Beams", info="Número de secuencias candidatas exploradas en paralelo durante la generación de texto mediante la búsqueda por haz. Un valor más alto aumenta las posibilidades de encontrar resultados coherentes y de alta calidad, pero puede ralentizar el proceso de generación.")
                tokens = gr.Slider(minimum=1, maximum=4096, step=1, value=512, label="Max new tokens", info="Limita la cantidad máxima de tokens generados en una sola iteración.")
                repet = gr.Slider(minimum=0.1, maximum=2.5, step=0.01, value=1.2, label="Repetition Penalty", info="Aplica una penalización a la probabilidad de tokens que ya se han generado, desalentando al modelo a repetir las mismas palabras o frases. La penalización se aplica dividiendo la probabilidad del token por un factor basado en el número de veces que ha aparecido el token en el texto generado.")        
                stream = gr.Checkbox(value=True, label="Stream")
                btnI = gr.Button("Send")
        
            btnI.click(interaction, inputs=[inp, temp, topp, topk, beams, tokens, repet, stream], outputs=chatbot)
          
    
    with gr.Tab("Get Audio"):
        
        gr.HTML("<div><p>Pegue el texto generado por guitarGPT y podrá reproducir el audio obtenido y descargar el archivo musicXML correspondiente</p></div>")
        
        
        with gr.Row():
            with gr.Column():
                out = gr.Textbox(label="Output")
                with gr.Row():
                    with gr.Column():
                        midiOutput = gr.File("/content/guitarGPT/utils/guitarGPT.mid", label="midi", elem_id='fileInput')
                    with gr.Column():
                        xmlOutput = gr.File("/content/guitarGPT/utils/guitarGPT.xml", label="xml")
                    with gr.Column():
                        playOutput = gr.Audio("/content/guitarGPT/utils/guitarGPT.wav")
            with gr.Column():
                inp = gr.Textbox(lines=5, label="Input", placeholder="*48 16D[5] 16K[4] 16I[4] |\n*48 16H[4] 16F[4] 16D[4] |\n*48 16F[4] 16D[4] 16C[3] |\n*48 48D[4] |\n... ")
                btnO = gr.Button("Send")
        
            btnO.click(getAudio, inp, [out, midiOutput, xmlOutput, playOutput])

demo.queue().launch(share=True, inbrowser=True, debug=False)
#demo.launch()
