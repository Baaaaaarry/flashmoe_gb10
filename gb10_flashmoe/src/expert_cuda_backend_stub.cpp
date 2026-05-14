#include "flashmoe/expert_cuda_backend.h"

#include <stdexcept>

namespace flashmoe {

bool cuda_expert_backend_available() {
    return false;
}

double cuda_unpack_and_execute_expert(const ExpertRecord&,
                                      DenseOperatorState&,
                                      const DenseRuntimeArtifact*,
                                      MaterializedExpertMap&,
                                      double*,
                                      double*,
                                      double*,
                                      float) {
    throw std::runtime_error("CUDA expert backend not available in this build");
}

void cuda_release_expert_buffers(MaterializedExpert&) {}

}  // namespace flashmoe
