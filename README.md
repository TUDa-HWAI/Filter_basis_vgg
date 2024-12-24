# Filter_basis_quant_vgg
This code Tests filter basis and bitwise shift on vgg16 and cifar100. You can set the size of filter basis and choose whether to use bitwise shift. Additionally, you can set bits for quantization.

# Train

python train.py --filter_basis_channel 1 --filter_basis_number 5 --filter_basis_weights_quant_bits 8 --coefficients_quant_bits 8 --activations_quant_bits 8 --coefficients_shift_en 0
