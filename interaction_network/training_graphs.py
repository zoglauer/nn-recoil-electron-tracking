import matplotlib.pyplot as plt
import argparse
import torch

def plot_losses(loss_dict, title):
    ax = plt.gca()
    ax.set_title(title)
    ax.plot(loss_dict["train_loss"], label='Train Loss')
    ax.plot(loss_dict["test_loss"], label='Test Loss')
    #ax.plot(loss_dict["test_acc"], label='Test Accuracy')
    ax.legend()
    plt.yscale("log")
    plt.savefig(f"{title}.png")

def main():
    parser = argparse.ArgumentParser(description='PyG Interaction Network Implementation')
    parser.add_argument('--model-save', type=str, default='./modelv2',
                        help='model save')
    parser.add_argument('--data-set', type=str, default='../data/RecoilElectrons.10k.data',
                        help='data set')
    args = parser.parse_args()

    loss_dict = torch.load(f"{args.model_save}/loss_dict.bin")

    plot_losses(loss_dict, title=args.model_save)


if __name__ == '__main__':
    main()