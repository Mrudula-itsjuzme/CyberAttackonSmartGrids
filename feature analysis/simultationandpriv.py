# Script 1: Simulation, Data Augmentation, and Privacy Techniques

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class SimulationPrivacy:
    def __init__(self, data):
        self.data = data

    def add_adversarial_noise(self, epsilon=0.1):
        print("Adding adversarial noise to simulate attacks...")
        noise = epsilon * np.sign(np.random.randn(*self.data.shape))
        adversarial_data = self.data + noise
        return adversarial_data

    def augment_data(self, augmentation_factor=3):
        print("Augmenting data with transformations...")
        augmented_data = []
        for _ in tqdm(range(augmentation_factor), desc="Data Augmentation"):
            noisy_data = self.add_adversarial_noise(epsilon=np.random.uniform(0.01, 0.1))
            augmented_data.append(noisy_data)
        return np.vstack(augmented_data)

    def apply_differential_privacy(self, sensitivity=1.0, epsilon=1.0):
        print("Applying differential privacy using Laplace noise...")
        laplace_noise = np.random.laplace(scale=sensitivity / epsilon, size=self.data.shape)
        private_data = self.data + laplace_noise
        return private_data

    def compare_with_original(self, transformed_data, title):
        differences = transformed_data - self.data
        mean_diff = np.mean(differences)
        max_diff = np.max(differences)

        print(f"{title} Summary:")
        print(f"Mean Difference: {mean_diff:.4f}")
        print(f"Max Difference: {max_diff:.4f}")

        plt.figure(figsize=(10, 6))
        plt.plot(self.data[0, :], label="Original", linestyle="--")
        plt.plot(transformed_data[0, :], label="Transformed", linestyle="-")
        plt.title(f"Comparison: {title}")
        plt.xlabel("Features")
        plt.ylabel("Values")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    print("Running Script 1: Simulation and Privacy Techniques")
    data = np.random.rand(1000, 10)  # Example data
    simulator = SimulationPrivacy(data)

    # Add adversarial noise
    adversarial_data = simulator.add_adversarial_noise()
    simulator.compare_with_original(adversarial_data, "Adversarial Noise")

    # Augment data
    augmented_data = simulator.augment_data()
    simulator.compare_with_original(augmented_data[:1000, :], "Augmented Data")

    # Apply differential privacy
    private_data = simulator.apply_differential_privacy()
    simulator.compare_with_original(private_data, "Differential Privacy")

    print("Script 1 completed: Simulation and Privacy Techniques")
