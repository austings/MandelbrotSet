from __future__ import print_function, division, absolute_import

import numpy as np
from timeit import default_timer as timer

from matplotlib.pylab import imshow, show
from numba import cuda


@cuda.jit(device=True)
def mandelbrot(x,y,max_iters):
    i = 0
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z+c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return 255


@cuda.jit
def create_fractal(min_x,max_x,min_y,max_y,image,iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    x,y = cuda.grid(2)

    if x < width and y < height:
        real = min_x + x*pixel_size_x
        imag = min_y + y * pixel_size_y
        color = mandelbrot(real, imag, iters)
        image[y, x] = color



start = timer()
imageY = 6400
imageX = 4800
image = np.zeros((imageY,imageX),dtype=np.uint8)

pixels = imageX * imageY
nThreads = 32
nBlocksX = imageX//nThreads + 1
nBlocksY = imageY//nThreads + 1
s = timer()
create_fractal[(nBlocksX, nBlocksY), (nThreads, nThreads)](-2.0, 1.0, -1.0, 1.0, image, 20)
e = timer()
m_time = e-s

print("This set took %f seconds" % m_time)
imshow(image)
show()


'''main()


NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
  
  The warning above means we are passing a variable into the CUDA kernel that cannot be accessed
  
  '''