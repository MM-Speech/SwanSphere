from torch.distributions import LogisticNormal


class LogitNormalTrainingTimesteps:
    def __init__(self, T=1000.0, loc=0.0, scale=1.0):
        assert T > 0
        self.T = T
        self.dist = LogisticNormal(loc, scale)

    def sample(self, size, device):
        t = self.dist.sample(size)[..., 0].to(device)
        return t


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch

    sampler = LogitNormalTrainingTimesteps()
    sampled_data = sampler.sample([100], 'cpu').numpy()

    # Plot the distribution of sampled data
    plt.figure(figsize=(8, 6))
    plt.hist(sampled_data, bins=30, density=True, alpha=0.6, color='b')
    plt.title("Logistic Normal Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()
