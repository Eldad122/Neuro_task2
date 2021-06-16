import numpy as np
import matplotlib.pyplot as plt
import kohonen

# First sample - Uniform Distribiotions
if __name__ == '__main__':
    x = np.random.uniform(-1000, 1000, 50)
    y = np.random.uniform(-1000, 1000, 50)
    step_max = 3000
    radius = 10
    alpha = 0.3
    kohonen.draw(x, y, 1, radius, alpha)
    kohonen.train(x, y, step_max, radius, alpha)


# Non-Uniform Distribiotions When using many neurons
def rand_non_uniform():
    x1 = np.random.uniform(-1000, 0, 25)
    y1 = np.random.uniform(0, 1000, 25)

    x2 = np.random.uniform(-1000, 0, 25)
    y2 = np.random.uniform(-1000, 0, 25)

    x3 = np.random.uniform(0, 1000, 25)
    y3 = np.random.uniform(-1000, 0, 25)

    x4 = np.random.uniform(0, 1000, 25)
    y4 = np.random.uniform(0, 1000, 25)

    x = np.append(x1, x2)
    x = np.append(x, x3)
    x = np.append(x, x4)

    y = np.append(y1, y2)
    y = np.append(y, y3)
    y = np.append(y, y4)
    return x, y


if __name__ == '__main__':
    x, y = rand_non_uniform()
    step_max = 2000
    radius = 200
    alpha = 0.4
    kohonen.draw(x, y, 1, radius, alpha)
    kohonen.train(x, y, step_max, radius, alpha)


# Second sample - Fitting a circle of neurons
def make_circle(s_range, e_range, amount=100):
    ang = np.random.uniform(0, 2 * np.pi, amount)
    r = np.random.uniform(s_range, e_range, amount)
    y = r * np.sin(ang)
    x = r * np.cos(ang)
    return x, y


if __name__ == '__main__':
    x, y = make_circle(500, 1000)
    step_max = 3000
    radius = 30
    alpha = 0.3
    kohonen.draw(x, y, 1, radius, alpha, False)
    x, y = kohonen.train(x, y, step_max, radius, alpha, is_line=False)
