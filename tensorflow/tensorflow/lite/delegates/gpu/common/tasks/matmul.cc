#include <string>
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {
std::string GetMatmulCode(const OperationDef& op_def) {
  std::vector<std::string> tensor_names(op_def.src_tensors.size());
  for (int i = 0; i < op_def.src_tensors.size(); ++i) {
    tensor_names[i] = "src_tensor_" + std::to_string(i);
  }//输入tensor


  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += "  int row = GLOBAL_ID_0;\n";
  c += "  int col = GLOBAL_ID_1;\n";
  const std::string col1 =
        "args." + tensor_names[0] + ".Width()"; 
  const std::string col2 =
        "args." + tensor_names[1] + ".Width()";  
  c +=  "  for (int i = 0; i < row; i++){\n"
  c +=	"    for (int j = 0; j < col;j++){\n"
  c +=	"		   float sum=0.0f;\n"
  c +=  "	     for (int k = 0; k < "+col1+"; k++)\n"
  c +=  "  		   sum += args." + tensor_names[0] + ".Read(k,i)*args."+tensor_names[1]+".Read(j,k);\n";
  c +=	"			 args.dst_tensor.Write(sum,j,i);\n";
  c +=  "  }}}\n";
  return c;
}

}  // namespace

GPUOperation CreateMatmul(const OperationDef& definition) {
  GPUOperation op(definition);
  for (int i = 0; i < definition.src_tensors.size(); ++i) {
    const std::string name = "src_tensor_" + std::to_string(i);
    op.AddSrcTensor(name, definition.src_tensors[i]);
  }
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.code_ = GetMatmulCode(definition);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
