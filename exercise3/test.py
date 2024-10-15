import numpy as np
import matplotlib.pyplot as plt

# RBF kernel function
def rbf_kernel(x1, x2, length_scale=1.0):
    sqdist = np.subtract.outer(x1, x2) ** 2
    return np.exp(-0.5 * sqdist / length_scale**2)

# Generate training data
X_train = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y_train = np.sin(X_train)

# Generate test points
X_test = np.linspace(0, 5, 100)

# Compute covariance matrices
K = rbf_kernel(X_train, X_train) + 1e-8 * np.eye(len(X_train))  # Add small noise for numerical stability
K_s = rbf_kernel(X_train, X_test)
K_ss = rbf_kernel(X_test, X_test)

# Compute the posterior mean and covariance
K_inv = np.linalg.inv(K)
mu_post = K_s.T @ K_inv @ y_train
cov_post = K_ss - K_s.T @ K_inv @ K_s

# Plotting the results
plt.figure(figsize=(10, 6))

# True function
plt.plot(X_test, np.sin(X_test), 'r', lw=2, label='True function')

# Posterior mean
plt.plot(X_test, mu_post, 'b', lw=2, label='GP mean')

# Confidence intervals (2 std dev)
std_post = np.sqrt(np.diag(cov_post))
plt.fill_between(X_test, mu_post - 2*std_post, mu_post + 2*std_post, color='blue', alpha=0.2, label='Confidence interval')

# Training points
plt.scatter(X_train, y_train, c='black', label='Training points')

plt.title('Gaussian Process with RBF Kernel')
plt.legend()
plt.show()
