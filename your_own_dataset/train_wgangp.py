import os
import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from fanogan.train_wgangp import train_wgangp

def main(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define transformations
    pipeline = [transforms.Resize([opt.img_size] * 2),
                transforms.RandomHorizontalFlip()]
    if opt.channels == 1:
        pipeline.append(transforms.Grayscale())
    pipeline.extend([transforms.ToTensor(),
                     transforms.Normalize([0.5] * opt.channels, [0.5] * opt.channels)])

    transform = transforms.Compose(pipeline)
    
    # Load training data (only real images)
    train_dataset = ImageFolder(os.path.join(opt.train_root, "real"), transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

    # Load test data (real and fake images)
    test_real_dataset = ImageFolder(os.path.join(opt.test_root, "real"), transform=transform)
    test_fake_dataset = ImageFolder(os.path.join(opt.test_root, "fake"), transform=transform)
    test_real_dataloader = DataLoader(test_real_dataset, batch_size=opt.batch_size, shuffle=False)
    test_fake_dataloader = DataLoader(test_fake_dataset, batch_size=opt.batch_size, shuffle=False)

    # Import Generator and Discriminator models
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from mvtec_ad.model import Generator, Discriminator

    # Initialize models
    generator = Generator(opt)
    discriminator = Discriminator(opt)

    # Train the model
    train_wgangp(opt, generator, discriminator, train_dataloader, device)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("train_root", type=str, help="root name of your dataset in train mode")
    parser.add_argument("test_root", type=str, help="root name of your dataset in test mode")
    parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
    parser.add_argument("--seed", type=int, default=None, help="value of a random seed")
    opt = parser.parse_args()

    main(opt)
