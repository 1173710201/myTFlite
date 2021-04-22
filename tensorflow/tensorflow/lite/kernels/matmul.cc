#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace custom {
namespace matmul {

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);//要求输入个数为2
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);//输出个数为1
  const TfLiteTensor* input1;
	const TfLiteTensor* input2;
	TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));//读取输入12
  TfLiteIntArray* input1_dims = input1->dims;//维度
  TfLiteIntArray* input2_dims = input2->dims;
  int input1_dims_size = input1_dims->size;
  int input2_dims_size = input2_dims->size;//输入tensor几维，12一样

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));//定义输出
  // Resize the output tensor.
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(input1_dims_size);
  //假设tensor都为2维矩阵
	output_shape->data[0] = input1_dims->data[0];//input1的行
	output_shape->data[1] = input2_dims->data[1];//input2的列

  output->type = input->type;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_shape));

  return kTfLiteOk;
}
template <typename T>
void mul(const T* in1,const T* in2,T* out,const int row,const int col,
	const int batch,const int mid){

  for (int b = 0; b < batch; b++) {
    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; ++j) {
      	int sum=0;
      	for (int k = 0; k < mid; k++){
      		sum+=in1[i*mid+k]*in2[k*col+j];
      	}
      	out[i * col + j] = sum;

      }
    }
    out += row * col;
  }
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  const TfLiteTensor* input1;
	const TfLiteTensor* input2;
	TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));//读取输入12
                    
  const int num_output_dims = output->dims->size;
  int batch_size = 1;
  for (int i = 0; i < num_output_dims - 2; ++i) {
    batch_size *= output->dims->data[i];
  }
  const int row_size = output->dims->data[num_output_dims - 2];
  const int col_size = output->dims->data[num_output_dims - 1];
  const int mid=input1->dims->data[1];//input1列，input2行
  mul<T>(GetTensorData<T>(input1),GetTensorData<T>(input2),GetTensorData<T>(output),
  				row_size,col_size,batch_size,mid);
  return kTfLiteOk;
}
}  // namespace matmul 
TfLiteRegistration* Register_MATMUL() {
  static TfLiteRegistration r = {nullptr, nullptr, matmul::Prepare,
                                 matmul::Eval};
  return &r;
}
}  // namespace custom
}  // namespace ops
}  // namespace tflite
