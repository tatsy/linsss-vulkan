import os
import sys
import cv2
import numpy as np
from scipy.special import sph_harm

N_THETA_DIVIDES = 128
N_PHI_DIVIDES = 256


def main(filename):
    base, ext = os.path.splitext(filename)
    sph_file = base + '.sph'

    # Compute harmonics coefficients
    print('Computing harmonics coefficients. Please wait...')
    coeffs = np.zeros((9, 3))
    indices = []
    index = 0
    for l in range(3):
        for m in range(-l, l + 1):
            indices.append((l, m, index))
            index += 1

    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise Exception('Failed to load image: %s' % (filename))

    height, width, _ = image.shape
    for l, m, k in indices:
        value = np.zeros((3))
        for i in range(N_THETA_DIVIDES):
            for j in range(N_PHI_DIVIDES):
                cosTheta = 2.0 * (i + 0.5) / N_THETA_DIVIDES - 1.0
                theta = np.arccos(max(-1.0, min(cosTheta, 1.0)))
                phi = (2.0 * np.pi * (j + 0.5)) / N_PHI_DIVIDES

                u = int(phi / (2.0 * np.pi) * width)
                v = int(theta / np.pi * height)
                u = max(0, min(u, width - 1))
                v = max(0, min(v, height - 1))

                color = image[v, u, :]
                w = np.real(sph_harm(m, l, phi, theta))
                value += 4.0 * np.pi * w * color

        coeffs[k] = value / (float)(N_THETA_DIVIDES * N_PHI_DIVIDES)

    with open(sph_file, 'w') as fp:
        for c in coeffs:
            fp.write('{0:f} {0:f} {0:f}\n'.format(c[0], c[1], c[2]))

    printf('Finish!')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('[ usage ] python env2sph.py INPUT_IMAGE_HDR')
    else:
        main(sys.argv[1])
