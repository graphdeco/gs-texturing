import os

if __name__ == '__main__':
    print(f"Installing environment")
    
    # Install torch
    print(f"Installing torch")
    os.system(f"conda install -y pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia")

    os.system(f"python -c 'import torch;print(torch.cuda.is_available())'")
    os.system(f"conda install -y pytorch3d -c pytorch3d")

    # Install requirements
    print(f"Installing rest requirements")
    os.system(f"pip install plyfile tqdm")
    
    # Install submodules
    print(f"Installing Simple KNN")
    os.system(f"pip install submodules/simple-knn")
    
    print(f"Installing Rasterizer")
    os.system(f"pip install submodules/diff-gaussian-rasterization-texture")
    
    print(f"Installing Graphdeco viewer")
    os.system(f"pip install submodules/graphdecoviewer")
