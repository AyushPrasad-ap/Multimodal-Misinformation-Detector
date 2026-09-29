import numpy
import pandas
import PIL
import sklearn
import torch
import torchvision
import transformers
import yaml


def main() -> None:
    print("=" * 65)
    print("Multimodal Misinformation Detection")
    print("Environment Verification")
    print("=" * 65)

    print(f"Python-compatible PyTorch: {torch.__version__}")
    print(f"Torchvision:               {torchvision.__version__}")
    print(f"Transformers:              {transformers.__version__}")
    print(f"scikit-learn:              {sklearn.__version__}")
    print(f"Pandas:                    {pandas.__version__}")
    print(f"NumPy:                     {numpy.__version__}")
    print(f"Pillow:                    {PIL.__version__}")
    print(f"YAML:                      {yaml.__version__}")

    print(f"CUDA available:            {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version:              {torch.version.cuda}")
        print(f"GPU:                       {torch.cuda.get_device_name(0)}")

    print("=" * 65)
    print("Environment verification complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
