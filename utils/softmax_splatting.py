# Adapted from https://github.com/sniklaus/softmax-splatting

import torch
import torch.nn.functional as F

import cupy
import re

import todos
import pdb

kernel_Softsplat_updateOutput = '''
	extern "C" __global__ void kernel_Softsplat_updateOutput(
		const int n,
		const float* input,
		const float* flow,
		float* output
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		const int intN = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
		const int intC = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
		const int intY = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
		const int intX = ( intIndex                                                    ) % SIZE_3(output);

		float dblOutputX = (float) (intX) + VALUE_4(flow, intN, 0, intY, intX);
		float dblOutputY = (float) (intY) + VALUE_4(flow, intN, 1, intY, intX);

		int intNorthwestX = (int) (floor(dblOutputX));
		int intNorthwestY = (int) (floor(dblOutputY));
		int intNortheastX = intNorthwestX + 1;
		int intNortheastY = intNorthwestY;
		int intSouthwestX = intNorthwestX;
		int intSouthwestY = intNorthwestY + 1;
		int intSoutheastX = intNorthwestX + 1;
		int intSoutheastY = intNorthwestY + 1;

		float dblNorthwest = ((float) (intSoutheastX) - dblOutputX   ) * ((float) (intSoutheastY) - dblOutputY   );
		float dblNortheast = (dblOutputX    - (float) (intSouthwestX)) * ((float) (intSouthwestY) - dblOutputY   );
		float dblSouthwest = ((float) (intNortheastX) - dblOutputX   ) * (dblOutputY    - (float) (intNortheastY));
		float dblSoutheast = (dblOutputX    - (float) (intNorthwestX)) * (dblOutputY    - (float) (intNorthwestY));

		if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(output)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(output))) {
			atomicAdd(&output[OFFSET_4(output, intN, intC, intNorthwestY, intNorthwestX)], VALUE_4(input, intN, intC, intY, intX) * dblNorthwest);
		}

		if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(output)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(output))) {
			atomicAdd(&output[OFFSET_4(output, intN, intC, intNortheastY, intNortheastX)], VALUE_4(input, intN, intC, intY, intX) * dblNortheast);
		}

		if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(output)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(output))) {
			atomicAdd(&output[OFFSET_4(output, intN, intC, intSouthwestY, intSouthwestX)], VALUE_4(input, intN, intC, intY, intX) * dblSouthwest);
		}

		if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(output)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(output))) {
			atomicAdd(&output[OFFSET_4(output, intN, intC, intSoutheastY, intSoutheastX)], VALUE_4(input, intN, intC, intY, intX) * dblSoutheast);
		}
	} }
'''

kernel_Softsplat_updateGradInput = '''
	extern "C" __global__ void kernel_Softsplat_updateGradInput(
		const int n,
		const float* input,
		const float* flow,
		const float* gradOutput,
		float* gradInput,
		float* gradFlow
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		const int intN = ( intIndex / SIZE_3(gradInput) / SIZE_2(gradInput) / SIZE_1(gradInput) ) % SIZE_0(gradInput);
		const int intC = ( intIndex / SIZE_3(gradInput) / SIZE_2(gradInput)                     ) % SIZE_1(gradInput);
		const int intY = ( intIndex / SIZE_3(gradInput)                                         ) % SIZE_2(gradInput);
		const int intX = ( intIndex                                                             ) % SIZE_3(gradInput);

		float dblGradInput = 0.0;

		float dblOutputX = (float) (intX) + VALUE_4(flow, intN, 0, intY, intX);
		float dblOutputY = (float) (intY) + VALUE_4(flow, intN, 1, intY, intX);

		int intNorthwestX = (int) (floor(dblOutputX));
		int intNorthwestY = (int) (floor(dblOutputY));
		int intNortheastX = intNorthwestX + 1;
		int intNortheastY = intNorthwestY;
		int intSouthwestX = intNorthwestX;
		int intSouthwestY = intNorthwestY + 1;
		int intSoutheastX = intNorthwestX + 1;
		int intSoutheastY = intNorthwestY + 1;

		float dblNorthwest = ((float) (intSoutheastX) - dblOutputX   ) * ((float) (intSoutheastY) - dblOutputY   );
		float dblNortheast = (dblOutputX    - (float) (intSouthwestX)) * ((float) (intSouthwestY) - dblOutputY   );
		float dblSouthwest = ((float) (intNortheastX) - dblOutputX   ) * (dblOutputY    - (float) (intNortheastY));
		float dblSoutheast = (dblOutputX    - (float) (intNorthwestX)) * (dblOutputY    - (float) (intNorthwestY));

		if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(gradOutput)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(gradOutput))) {
			dblGradInput += VALUE_4(gradOutput, intN, intC, intNorthwestY, intNorthwestX) * dblNorthwest;
		}

		if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(gradOutput)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(gradOutput))) {
			dblGradInput += VALUE_4(gradOutput, intN, intC, intNortheastY, intNortheastX) * dblNortheast;
		}

		if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(gradOutput)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(gradOutput))) {
			dblGradInput += VALUE_4(gradOutput, intN, intC, intSouthwestY, intSouthwestX) * dblSouthwest;
		}

		if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(gradOutput)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(gradOutput))) {
			dblGradInput += VALUE_4(gradOutput, intN, intC, intSoutheastY, intSoutheastX) * dblSoutheast;
		}

		gradInput[intIndex] = dblGradInput;
	} }
'''

kernel_Softsplat_updateGradFlow = '''
	extern "C" __global__ void kernel_Softsplat_updateGradFlow(
		const int n,
		const float* input,
		const float* flow,
		const float* gradOutput,
		float* gradInput,
		float* gradFlow
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		float dblGradFlow = 0.0;

		const int intN = ( intIndex / SIZE_3(gradFlow) / SIZE_2(gradFlow) / SIZE_1(gradFlow) ) % SIZE_0(gradFlow);
		const int intC = ( intIndex / SIZE_3(gradFlow) / SIZE_2(gradFlow)                    ) % SIZE_1(gradFlow);
		const int intY = ( intIndex / SIZE_3(gradFlow)                                       ) % SIZE_2(gradFlow);
		const int intX = ( intIndex                                                          ) % SIZE_3(gradFlow);

		float dblOutputX = (float) (intX) + VALUE_4(flow, intN, 0, intY, intX);
		float dblOutputY = (float) (intY) + VALUE_4(flow, intN, 1, intY, intX);

		int intNorthwestX = (int) (floor(dblOutputX));
		int intNorthwestY = (int) (floor(dblOutputY));
		int intNortheastX = intNorthwestX + 1;
		int intNortheastY = intNorthwestY;
		int intSouthwestX = intNorthwestX;
		int intSouthwestY = intNorthwestY + 1;
		int intSoutheastX = intNorthwestX + 1;
		int intSoutheastY = intNorthwestY + 1;

		float dblNorthwest = 0.0;
		float dblNortheast = 0.0;
		float dblSouthwest = 0.0;
		float dblSoutheast = 0.0;

		if (intC == 0) {
			dblNorthwest = ((float) (-1.0)) * ((float) (intSoutheastY) - dblOutputY   );
			dblNortheast = ((float) (+1.0)) * ((float) (intSouthwestY) - dblOutputY   );
			dblSouthwest = ((float) (-1.0)) * (dblOutputY    - (float) (intNortheastY));
			dblSoutheast = ((float) (+1.0)) * (dblOutputY    - (float) (intNorthwestY));

		} else if (intC == 1) {
			dblNorthwest = ((float) (intSoutheastX) - dblOutputX   ) * ((float) (-1.0));
			dblNortheast = (dblOutputX    - (float) (intSouthwestX)) * ((float) (-1.0));
			dblSouthwest = ((float) (intNortheastX) - dblOutputX   ) * ((float) (+1.0));
			dblSoutheast = (dblOutputX    - (float) (intNorthwestX)) * ((float) (+1.0));

		}

		for (int intChannel = 0; intChannel < SIZE_1(gradOutput); intChannel += 1) {
			float dblInput = VALUE_4(input, intN, intChannel, intY, intX);

			if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(gradOutput)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(gradOutput))) {
				dblGradFlow += dblInput * VALUE_4(gradOutput, intN, intChannel, intNorthwestY, intNorthwestX) * dblNorthwest;
			}

			if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(gradOutput)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(gradOutput))) {
				dblGradFlow += dblInput * VALUE_4(gradOutput, intN, intChannel, intNortheastY, intNortheastX) * dblNortheast;
			}

			if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(gradOutput)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(gradOutput))) {
				dblGradFlow += dblInput * VALUE_4(gradOutput, intN, intChannel, intSouthwestY, intSouthwestX) * dblSouthwest;
			}

			if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(gradOutput)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(gradOutput))) {
				dblGradFlow += dblInput * VALUE_4(gradOutput, intN, intChannel, intSoutheastY, intSoutheastX) * dblSoutheast;
			}
		}

		gradFlow[intIndex] = dblGradFlow;
	} }
'''


def cupy_kernel(strFunction, objectVariables):
    strKernel = globals()[strFunction]

    while True:
        objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objectMatch is None:
            break

        intArg = int(objectMatch.group(2))

        strTensor = objectMatch.group(4)
        intSizes = objectVariables[strTensor].size()

        strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))

    while True:
        objectMatch = re.search('(OFFSET_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objectMatch is None:
            break

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objectMatch.group(0), '(' + str.join('+', strIndex) + ')')

    while True:
        objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objectMatch is None:
            break

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')

    return strKernel


@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)


class _FunctionSoftsplat(torch.autograd.Function):
    @staticmethod
    def forward(self, input, flow):
        # ==> pdb.set_trace()
        self.save_for_backward(input, flow)

        intSamples = input.shape[0]
        intInputDepth, intInputHeight, intInputWidth = input.shape[1], input.shape[2], input.shape[3]
        intFlowDepth, intFlowHeight, intFlowWidth = flow.shape[1], flow.shape[2], flow.shape[3]

        assert (intFlowDepth == 2)
        assert (intInputHeight == intFlowHeight)
        assert (intInputWidth == intFlowWidth)

        assert (input.is_contiguous() == True)
        assert (flow.is_contiguous() == True)

        output = input.new_zeros([intSamples, intInputDepth, intInputHeight, intInputWidth])

        if input.is_cuda == True:
            n = output.nelement()
            cupy_launch('kernel_Softsplat_updateOutput', cupy_kernel('kernel_Softsplat_updateOutput', {
                'input': input,
                'flow': flow,
                'output': output
            }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, input.data_ptr(), flow.data_ptr(), output.data_ptr()]
            )

        elif input.is_cuda == False:
            raise NotImplementedError()

        # todos.debug.output_var("input",input)
        # todos.debug.output_var("flow",flow)
        # todos.debug.output_var("output",output)
		# tensor [input] size: [1, 257, 224, 448], min: -1.265544, max: 10.693622, mean: 0.712642
		# tensor [flow] size: [1, 2, 224, 448], min: -248.74115, max: 0.575409, mean: -58.517563
		# tensor [output] size: [1, 257, 224, 448], min: -1.265544, max: 10.693622, mean: 0.712642

        return output

    @staticmethod
    def backward(self, gradOutput):
        pdb.set_trace()
        input, flow = self.saved_tensors

        intSamples = input.shape[0]
        intInputDepth, intInputHeight, intInputWidth = input.shape[1], input.shape[2], input.shape[3]
        intFlowDepth, intFlowHeight, intFlowWidth = flow.shape[1], flow.shape[2], flow.shape[3]

        assert (intFlowDepth == 2)
        assert (intInputHeight == intFlowHeight)
        assert (intInputWidth == intFlowWidth)

        assert (gradOutput.is_contiguous() == True)

        gradInput = input.new_zeros([intSamples, intInputDepth, intInputHeight, intInputWidth]) if \
            self.needs_input_grad[0] == True else None
        gradFlow = input.new_zeros([intSamples, intFlowDepth, intFlowHeight, intFlowWidth]) if self.needs_input_grad[
                                                                                                   1] == True else None

        if input.is_cuda == True:
            if gradInput is not None:
                n = gradInput.nelement()
                cupy_launch('kernel_Softsplat_updateGradInput', cupy_kernel('kernel_Softsplat_updateGradInput', {
                    'input': input,
                    'flow': flow,
                    'gradOutput': gradOutput,
                    'gradInput': gradInput,
                    'gradFlow': gradFlow
                }))(
                    grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                    block=tuple([512, 1, 1]),
                    args=[n, input.data_ptr(), flow.data_ptr(), gradOutput.data_ptr(), gradInput.data_ptr(), None]
                )

            if gradFlow is not None:
                n = gradFlow.nelement()
                cupy_launch('kernel_Softsplat_updateGradFlow', cupy_kernel('kernel_Softsplat_updateGradFlow', {
                    'input': input,
                    'flow': flow,
                    'gradOutput': gradOutput,
                    'gradInput': gradInput,
                    'gradFlow': gradFlow
                }))(
                    grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                    block=tuple([512, 1, 1]),
                    args=[n, input.data_ptr(), flow.data_ptr(), gradOutput.data_ptr(), None, gradFlow.data_ptr()]
                )

        elif input.is_cuda == False:
            raise NotImplementedError()

        return gradInput, gradFlow


def FunctionSoftsplat(tensorInput, tensorFlow, tensorMetric, output_size=None):
    assert (tensorMetric is None or tensorMetric.shape[1] == 1)
    tensorInput = torch.cat([tensorInput * tensorMetric, tensorMetric], 1)
    tensorOutput = _FunctionSoftsplat.apply(tensorInput, tensorFlow)
    # tensorOutput = tensorInput

    tenSplattedMetric = tensorOutput[:, -1:, :, :]
    tenSplattedMetric[tenSplattedMetric == 0] = 1
    tensorOutput = tensorOutput[:, :-1, :, :] / tenSplattedMetric

    tensorOutput = tensorOutput[:, :, :output_size[0], :output_size[1]]

	# tensor [FunctionSoftsplat: tensorInput] size: [1, 1, 888, 1776], min: 1.0, max: 1.0, mean: 1.0
	# tensor [FunctionSoftsplat: tensorFlow] size: [1, 2, 888, 1776], min: -888.0, max: 99.450378, mean: -212.031052
	# tensor [FunctionSoftsplat: tensorMetric] size: [1, 1, 888, 1776], min: 0.0, max: 1.0, mean: 0.5
	# FunctionSoftsplat: output_size is tuple: len = 2
	#     [item] value: '888'
	#     [item] value: '888'
	# tensor [FunctionSoftsplat: tensorOutput] size: [1, 1, 888, 888], min: 1.0, max: 1.0, mean: 1.0
	# --------------------------------------------------------------------------------
	# tensor [FunctionSoftsplat: tensorInput] size: [1, 32, 1780, 3560], min: -1.075294, max: 3.646587, mean: 0.008901
	# tensor [FunctionSoftsplat: tensorFlow] size: [1, 2, 1780, 3560], min: -1780.0, max: 198.96228, mean: -424.981659
	# tensor [FunctionSoftsplat: tensorMetric] size: [1, 1, 1780, 3560], min: 0.0, max: 1.0, mean: 0.5
	# FunctionSoftsplat: output_size is tuple: len = 2
	#     [item] value: '1780'
	#     [item] value: '1780'
	# tensor [FunctionSoftsplat: tensorOutput] size: [1, 32, 1780, 1780], min: -1.075294, max: 3.646587, mean: 0.008901
	# --------------------------------------------------------------------------------

    return tensorOutput




def pytorch_softsplat(input_tensor, flow):
    batch_size, channels, height, width = input_tensor.shape
    
    # 将输入和光流调整为正确的形状
    input_tensor = input_tensor.view(batch_size, channels, -1)
    flow = flow.view(batch_size, 2, -1)
    
    # 为每个像素生成坐标
    coords_y, coords_x = torch.meshgrid(torch.arange(height), torch.arange(width))
    coords = torch.stack([coords_x.float(), coords_y.float()], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    # 应用光流
    coords = coords + flow
    
    # 将坐标归一化到[0, 1]区间
    coords[:, 0, :, :] = (coords[:, 0, :, :] / (width - 1)) * 2.0 - 1.0
    coords[:, 1, :, :] = (coords[:, 1, :, :] / (height - 1)) * 2.0 - 1.0
    
    # 使用grid_sample进行双线性插值
    output_tensor = F.grid_sample(input_tensor, coords, mode='bilinear', padding_mode='zeros')
    
    return output_tensor.view(batch_size, channels, height, width)

