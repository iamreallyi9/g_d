from nni.compression.torch import ModelSpeedup
from nni.compression.torch import SlimPruner
from gj_hourglass import HourglassModel
from nni.compression.torch import apply_compression_results
import torch
from torchsummaryX import summary
import time

def test_nni():
    model = load_t_net()

    config_list = [{
        'sparsity': 0.5,
        'op_types': ['Conv2d']
    }]
    pruner = SlimPruner(model, config_list)
    model = pruner.compress()

    print(model)
    masks_file = "./nni/mask.pth"
    pruner.export_model(model_path="./nni/nni_mod.pth", mask_path=masks_file)
    print("export ok")
    apply_compression_results(model, masks_file)

    # model: 要加速的模型
    # dummy_input: 模型的示例输入，传给 `jit.trace`
    # masks_file: 剪枝算法创建的掩码文件
    dummy_input = torch.randn(1,3,384,224)
    m_speedup = ModelSpeedup(model, dummy_input.cuda(), masks_file)
    m_speedup.speedup_model()
    dummy_input = dummy_input.cuda()
    start = time.time()
    out = model(dummy_input)
    summary(model,dummy_input)
    print('elapsed time: ', time.time() - start)



def load_t_net():
    new_model = HourglassModel(3)
    new_model = torch.nn.DataParallel(new_model)
    model_file = "results/ayush/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS4_Oadam/checkpoints/0020.pth"
    model_parameters = torch.load(model_file)
    new_model.load_state_dict(model_parameters)
    return new_model.module

if __name__ == '__main__':
    test_nni()