BASE_MODEL="yahma/llama-7b-hf"
LORA_PATH="Chinese-Vicuna/Chinese-Vicuna-lora-7b-chatv1"
INT8=1
SHOW_BEAM=0 # 是否显示全部的beam生成效果
if [[ $DEBUG -eq 1 ]]
then
CUDA_VISIBLE_DEVICES=0 jurigged -v chat.py --model_path $BASE_MODEL --lora_path $LORA_PATH\
    --int8 $INT8\
    --use_typewriter 1 \
    --show_beam $SHOW_BEAM \
    --prompt_type "chat" \
    --share_link 0 
else
CUDA_VISIBLE_DEVICES=0 python chat.py --model_path $BASE_MODEL --lora_path $LORA_PATH\
    --int8 $INT8\
    --use_typewriter 1 \
    --show_beam $SHOW_BEAM \
    --prompt_type "chat" \
    --share_link 0 
fi