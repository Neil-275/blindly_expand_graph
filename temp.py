import numpy as np


def fake_embed(text: np.array):
    return np.random.rand(4)


if __name__ == '__main__':
    a = np.array([
        "Hello",
        "Welcome to the world of Python programming!"
    ])
    embeddings = np.apply_along_axis(
        fake_embed,
        1,
        a.reshape(-1, 1)
    )

    print(embeddings)
