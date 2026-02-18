def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    for _ in range(steps):
        f_grad = 2*a*x0+b
        x0 -= lr * f_grad
    return x0