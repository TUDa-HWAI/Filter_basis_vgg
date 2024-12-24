# Filter_basis_quant_vgg
This code tests filter basis and bitwise shift on VGG16 and CIFAR-100. You can configure the size of the filter basis and decide whether to use bitwise shift. Additionally, you can set the number of bits for quantization. To enable bitwise shift for linear combination coefficients, set `coefficients_shift_en` to 1. In this case, the value of `coefficients_quant_bits` will no longer be relevant.

# Train

python train.py --filter_basis_channel 1 --filter_basis_number 5 --filter_basis_weights_quant_bits 8 --coefficients_quant_bits 8 --activations_quant_bits 8 --coefficients_shift_en 0
