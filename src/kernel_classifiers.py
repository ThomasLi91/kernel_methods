import numpy as np
import cvxpy as cp


class SVM:
    def __init__(self, kernel, C=1.0):
        self.kernel_fn = kernel
        self.C = C
        self.alpha = None
        self.bias = None
        self.epsilon = 1e-5
        self.support_vectors = None
        self.kernel_matrix = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        kernel_matrix = self.kernel_fn(X, X)

        # Variables of the dual problem
        alpha = cp.Variable(n_samples)
        y = y.astype(float)

        # Objective function
        objective = cp.Maximize(
            cp.sum(alpha)
            - 0.5 * cp.quad_form(cp.multiply(y, alpha), kernel_matrix, assume_PSD=True)
        )

        # Constraints
        constraints = [alpha >= 0, alpha <= self.C, cp.sum(cp.multiply(y, alpha)) == 0]

        # Solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # Extract support vectors
        support_vector_indices = np.where(
            (alpha.value > self.epsilon) & (alpha.value < self.C - self.epsilon)
        )[0]
        non_zero_indices = np.where(alpha.value > self.epsilon)[0]
        self.alpha = alpha.value[non_zero_indices]
        self.support_vectors = X[non_zero_indices]
        self.y_train = y[non_zero_indices]

        # Extract bias
        Ka = kernel_matrix @ (alpha.value * y)
        self.bias = np.mean(y[support_vector_indices] - Ka[support_vector_indices])

    def predict(self, X):
        kernel_matrix = self.kernel_fn(X, self.support_vectors)
        decision_function = (
            np.dot(kernel_matrix, (self.alpha * self.y_train)) + self.bias
        )

        return np.sign(decision_function)