from datetime import datetime
import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from OmniAvatar.utils.args_config import parse_args
args = parse_args()

from tqdm import tqdm
from OmniAvatar.models.training_module import OmniTrainingModule
from scripts.inference import read_from_file, set_seed

def build_inputs(trainer_module): 
    # 读取inference输入
    set_seed(args.seed)
    data_iter = read_from_file(args.input_file)
    exp_name = os.path.basename(args.exp_path)
    date_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f'demo_out/{exp_name}/res_{os.path.splitext(os.path.basename(args.input_file))[0]}_'\
                f'seed{args.seed}_step{args.num_steps}_{args.fps}_{date_name}'
    for idx, text in tqdm(enumerate(data_iter)):
        if len(text) == 0:
            continue
        input_list = text.split("@@")
        assert len(input_list)<=3
        if len(input_list) == 0:
            continue
        elif len(input_list) == 1:
            text, image_path, audio_path = input_list[0], None, None
        elif len(input_list) == 2:
            text, image_path, audio_path = input_list[0], input_list[1], None
        elif len(input_list) == 3:
            text, image_path, audio_path = input_list[0], input_list[1], input_list[2]
        audio_dir = output_dir + '/audio'
        os.makedirs(audio_dir, exist_ok=True)
        input_audio_path = audio_path
        prompt_dir = output_dir + '/prompt'
        os.makedirs(prompt_dir, exist_ok=True)
    
    
    data = {
        "prompt": text,
        "image_path": image_path,
        "audio_path": input_audio_path,
        "output_dir": output_dir,
    }
    return trainer_module.sample_preprocess(data)

def main():
    device = torch.device("cuda:0")
    module = OmniTrainingModule(args)
    module.to(device)
    module.pipe.device = device      # 让内部 pipeline 也指向 GPU
    module.pipe.load_models_to_device(["vae", "dit"])
    module.eval()

    ckpt = args.infer_ckpt_path
    state = torch.load(ckpt)  # 如果是 {"module": ...} 取里面那层
    if "state_dict" in state:
        state = state["state_dict"]
    module.load_state_dict(state, strict=False)
    
    inputs = build_inputs(module)
    module.sample_video(inputs)


# import debugpy
# debugpy.listen(5678)
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()
if __name__ == "__main__":
    main()