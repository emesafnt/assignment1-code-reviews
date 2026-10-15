"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# Multiply each row by each column.
# AB = [[3*1+4*0, 3*2+4*(-1)], [2*1+1*0, 2*2+1*(-1)]].
# BA = [[1*3+2*2, 1*4+2*1], [0*3-1*2, 0*4-1*1]].
# AB = [[3, 2], [2, 3]], BA = [[7, 6], [-2, -1]].
# A+B = [[4, 6], [2, 0]], so (A+B).T = [[4, 2], [6, 0]].
# B.T = [[1, 0], [2, -1]], so AB.T = [[11, -4], [4, -1]].
# trace(A) = 3+1 = 4; trace(AB) = 3+3 = 6.
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    A = np.array([[3, 4], [2, 1]] if A is None else A, dtype=float)
    B = np.array([[1, 2], [0, -1]] if B is None else B, dtype=float)
    AB = A @ B
    return AB, B @ A, (A + B).T, A @ B.T, np.trace(A), np.trace(AB)


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# det(M-lambda*I) = (2-lambda)**2-1 = (lambda-1)*(lambda-3).
# The eigenvalues are 1 and 3.
# For lambda=1, x=y, so take v=[1,1]. Mv=[1,1]=1*v.
# For lambda=3, x=-y, so take v=[1,-1]. Mv=[3,-3]=3*v.
# Divide each vector by sqrt(2) to make its length 1.
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    M = np.array([[2, -1], [-1, 2]] if M is None else M, dtype=float)
    return np.linalg.eigh(M)


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# Let A have shape (m,n) and B have shape (n,m).
# trace(AB) = sum_i sum_j A[i,j]*B[j,i].
# Switch the order of the finite sums and the scalar factors:
# = sum_j sum_i B[j,i]*A[i,j] = trace(BA).

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# det(C) = 4*6-7*2 = 10. It is nonzero, so the inverse exists.
# For a 2x2 inverse, swap the diagonal entries and negate the other two.
# C inverse = (1/10)*[[6,-7],[-2,4]] = [[0.6,-0.7],[-0.2,0.4]].
# C*[[6,-7],[-2,4]] = [[10,0],[0,10]], which verifies the inverse.
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    C = np.array([[4, 7], [2, 6]] if C is None else C, dtype=float)
    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# The exponential series gives sum_k rate**k/k! = exp(rate).
# Total probability is c*exp(rate)=1, so c=exp(-rate).
# E[X] = c*sum_{k>=1} k*rate**k/k!
#      = c*rate*sum_{j>=0} rate**j/j! = rate.
# E[X*(X-1)] = c*rate**2*sum_{j>=0} rate**j/j! = rate**2.
# E[X**2] = rate**2+rate, so Var(X)=rate.
# With rate=3: c=exp(-3), mean=3, variance=3.
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    return np.exp(-rate), float(rate), float(rate)


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# P(A intersect B) = P(A|B)*P(B) = 0.6*0.3 = 0.18.
# P(B|A) = P(A intersect B)/P(A) = 0.18/0.5 = 0.36.
# P(A union B) = P(A)+P(B)-P(A intersect B) = 0.62.
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    intersection = p_a_given_b * p_b
    return intersection, intersection / p_a, p_a + p_b - intersection


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# There are 3 heads and 2 tails. L(theta)=theta**3*(1-theta)**2.
# log L = 3*log(theta)+2*log(1-theta).
# Its derivative is 3/theta-2/(1-theta)=0.
# Thus 3*(1-theta)=2*theta, giving MLE=3/5.
# Multiply by the prior 2*theta: posterior is proportional to
# 2*theta**4*(1-theta)**2. Its log derivative is
# 4/theta-2/(1-theta)=0, giving MAP=4/6=2/3.
# Both log second derivatives are negative inside (0,1).
# In general MLE=heads/tosses and MAP=(heads+1)/(tosses+1).
# With no heads the MLE is 0; with all heads both estimates are 1.
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    return heads / tosses, (heads + 1) / (tosses + 1)


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# Write v=sigma**2 and n=len(data), assuming independent samples.
# L = (2*pi*v)**(-n/2)*exp(-sum((x_i-mu)**2)/(2*v)).
# log L = -n/2*log(2*pi*v)-sum((x_i-mu)**2)/(2*v).
# d(log L)/d(mu) = sum(x_i-mu)/v = 0, so mu=sum(x_i)/n.
# d(log L)/d(v) = -n/(2*v)+sum((x_i-mu)**2)/(2*v**2)=0.
# Thus v=sum((x_i-mu)**2)/n. Use n, not n-1, for this MLE.
# The mean minimizes the squared errors. At that mean, the log likelihood
# increases until v equals their average and decreases afterwards.
# For [1,3,5,7], mu=16/4=4 and v=(9+1+1+9)/4=5.
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    data = np.array([1, 3, 5, 7] if data is None else data, dtype=float)
    mean = np.mean(data)
    variance = np.mean((data - mean) ** 2)
    return mean, variance


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# log posterior = constant-(x-mu)**2/(2*observation_variance)
#                         -(mu-prior_mean)**2/(2*prior_variance).
# Set the derivative to zero:
# (x-mu)/observation_variance+(prior_mean-mu)/prior_variance=0.
# mu = (x/observation_variance+prior_mean/prior_variance)
#      / (1/observation_variance+1/prior_variance).
# The second derivative is negative, so this is the posterior mode.
# Here mu=(5/4+0)/(1/4+1)=1.
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    return ((x / observation_variance + prior_mean / prior_variance)
            / (1 / observation_variance + 1 / prior_variance))


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# Let m=E[X], assuming E[X**2] is finite.
# Var(X)=E[(X-m)**2]=E[X**2-2*m*X+m**2].
# = E[X**2]-2*m*E[X]+m**2 = E[X**2]-m**2.

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# Treat X and m as column vectors. Let m=E[X].
# Sigma=E[(X-m)@(X-m).T] is the covariance matrix.
# Entry (i,j) is E[(X_i-m_i)*(X_j-m_j)]. Assume finite second moments.
# E[u.T@X]=u.T@m because u is fixed.
# Var(u.T@X)=E[(u.T@(X-m))**2]
# = E[u.T@(X-m)@(X-m).T@u]
# = u.T@E[(X-m)@(X-m).T]@u = u.T@Sigma@u.
# A unit u makes this the variance along a direction of length 1.

# P3 Q2: Describe the Gaussian generator and observed correlation.
# Draw 500 pairs from a Gaussian with mean [2,-1] by default.
# The covariance is [[3,1.2],[1.2,1]]. The population correlation is
# 1.2/sqrt(3*1), about 0.693, so the scatter should slope upward.
# With the default seed, the sample correlation is about 0.699.
# The points form an upward-sloping cloud: larger values tend to occur together.
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    mean = np.array([2, -1] if mean is None else mean, dtype=float)
    covariance = np.array([[3, 1.2], [1.2, 1]] if covariance is None
                          else covariance, dtype=float)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=n_samples)


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# Subtract the sample mean so each feature is centered at zero.
# Estimate Sigma using centered.T@centered/(n-1).
# Each unit eigenvector u has projection variance u.T@Sigma@u=lambda.
# As in Lecture 3, maximize u.T@Sigma@u with u.T@u=1.
# L=u.T@Sigma@u-lambda*(u.T@u-1) gives 2*Sigma@u-2*lambda*u=0.
# So Sigma@u=lambda*u; the largest eigenvalue gives the most variance.
# Lecture 3 uses divisor n. The template requires n-1, which rescales the
# eigenvalues but leaves the directions and variance fractions unchanged.
# Sort eigenvalues from largest to smallest to put the main direction first.
# Scores are projections of the centered samples onto these directions.
# The plot draws both directions through the mean, with lengths scaled
# by the square root of each eigenvalue to show their different spreads.
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=0)
    centered = X - mean
    covariance = centered.T @ centered / (len(X) - 1)
    eigenvalues, components = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    # Roundoff can make a zero eigenvalue slightly negative.
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = components[:, order]
    return {"mean": mean, "centered": centered, "covariance": covariance,
            "eigenvalues": eigenvalues, "components": components,
            "scores": centered @ components}


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# Divide each eigenvalue by their sum to get its variance fraction.
# The default population covariance has eigenvalues about 3.562 and 0.438,
# giving about 89.1% and 10.9%. Our default sample gives 89.4% and 10.6%.
# To keep only PC1, use scores[:, :1] as the one-dimensional dataset.
# Keeping PC1 reduces two features to one while keeping most of the variance.
# It loses the variation along PC2. Keeping both loses no dimensions.
# High explained variance does not guarantee that all useful information
# is kept; a low-variance direction can still matter for a later task.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return eigenvalues / np.sum(eigenvalues)


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# The first panel shows the samples and both principal directions.
# The second shows how much variance each direction explains.
# Equal axis scaling keeps the directions and spreads visually accurate.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    import matplotlib.pyplot as plt

    fractions = explained_variance(result["eigenvalues"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(X[:, 0], X[:, 1], s=12, alpha=0.4, label="Samples")
    mean = result["mean"]
    for i, color in enumerate(["tab:red", "tab:green"]):
        direction = result["components"][:, i]
        offset = 2 * np.sqrt(result["eigenvalues"][i]) * direction
        ends = np.array([mean - offset, mean + offset])
        axes[0].plot(ends[:, 0], ends[:, 1], color=color, linewidth=2,
                     label=f"PC{i + 1}")
    axes[0].scatter(*mean, color="black", s=30, label="Mean")
    axes[0].set(xlabel="Feature 1", ylabel="Feature 2",
                title="Gaussian data and principal directions")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend()
    axes[1].bar(["PC1", "PC2"], fractions * 100,
                color=["tab:red", "tab:green"])
    for i, fraction in enumerate(fractions):
        axes[1].text(i, fraction * 100 + 1, f"{fraction:.1%}", ha="center")
    axes[1].set(xlabel="Principal component", ylabel="Explained variance (%)",
                title="Variance kept by each component", ylim=(0, 110))
    fig.tight_layout()
    return fig


# REFLECTION: contributions, tasks completed, and external resources used.
# Completed the derivations, matrix calculations, probability estimates,
# Gaussian data generation, PCA, and plots.
# References: assignment README and Lectures 3, 4, and 5.
# Lecture 3 provides the linear algebra and PCA methods used here.


if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    import matplotlib.pyplot as plt

    print("Matrix operations:", matrix_operations())
    print("Eigenpairs:", eigensystem())
    print("Inverse:", matrix_inverse())
    print("Distribution moments:", distribution_moments())
    print("Event probabilities:", event_probabilities())
    print("Coin MLE and MAP:", coin_estimates())
    print("Gaussian MLE:", gaussian_mle())
    print("Gaussian MAP:", gaussian_map())
    X = generate_data()
    result = pca(X)
    print("Sample correlation:", np.corrcoef(X, rowvar=False)[0, 1])
    print("Explained variance:", explained_variance(result["eigenvalues"]))
    fig = make_plots(X, result)
    fig.savefig("Assignment1_pca.png", dpi=180, bbox_inches="tight")
    plt.show()
