![AlexNet](https://miro.medium.com/max/960/0*pJ3o_2zTTNnixhKH.png)

| **Layer** | **Input size** | **Kernel**| **Padding** | **Stride**|**Formula**| **Output size**|
|-----------|----------------|-----------|-------------|-----------|-----------|----------------|
|Conv 1     |  224 * 224 * 3 | 11 * 11 * 96| [1, 2]    | 4         | $\frac{[224-11+(1+2)]}{4}+1$|55 * 55 * 96|
|Polling 1  | 55 * 55 * 96   | 3 * 3     |      0      |  2        | $\frac{55-3}{2}+1$|27 * 27 * 96|
|Conv 2     |  27 * 27 * 96 | 5 * 5 * 256|     [2, 2]  |  1         | $\frac{[27-5+(2+2)]}{1}+1$|27 * 27 * 256|
|Polling 2  | 55 * 55 * 96   | 3 * 3     |      0      |  2        | $\frac{27-3}{2}+1$|13 * 13 * 256|
|Conv 3     |  13 * 13 * 256 | 3 * 3 * 384|     [1, 1]  |  1         | $\frac{[13-3+(1+1)]}{1}+1$|13 * 13 * 384|
|Conv 4     |  13 * 13 * 384 | 3 * 3 * 384|     [1, 1]  |  1         | $\frac{[13-3+(1+1)]}{1}+1$|13 * 13 * 384|
|Conv 5     |  13 * 13 * 384 | 3 * 3 * 256|     [1, 1]  |  1         | $\frac{[13-3+(1+1)]}{1}+1$|13 * 13 * 256|
|Polling 3  | 13 * 13 * 256  | 3 * 3     |      0      |  2        | $\frac{13-3}{2}+1$|6 * 6 * 256|
|FC 1       | 6 * 6 * 256 = 9216|        |             |            |          | 4096 |
|FC 2       |       4096     |           |             |            |          | 4096 |
|FC 3       |       4096     |           |             |            |          | 1000 |


# Pytorch tips

* Use `nn.Sequential` to build the model.

* Pytorch在计算conv和pooling的output size时，都是向下取整

* The first convolutional layer has the padding with (1, 2), i.e. one row and column of zeros at the top and left of the image and two rows and columns at the right and under of the image. But in Pytorch building parameters in nn.Conv2d()， there is only two types of `padding=`
    1. padding = 2, means 2 padding circles around data
    2. padding = (1, 2), means one rows of zeros above and below the image, and two columns of zeros left and right of the image.
  Therefore facing the problem of padding (1, 2), we need to use `nn.ZeroPad2d((1, 2, 1, 2))`.