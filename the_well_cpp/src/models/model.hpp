// ============================================================================
// MODEL.HPP â€” Base class for surrogate PDE models
//
// All models implement:
//   forward()  â€” predict next timestep from history
//   backward() â€” compute gradients w.r.t. parameters
//   params()   â€” return parameter/gradient tensor pairs
//
// Input:  [batch, history, N]  (history timesteps of N-node field)
// Output: [batch, N]           (predicted next timestep)
// ============================================================================
#pragma once

#include "tensor.hpp"

namespace well {

// Parameter with its gradient
struct Param {
    Tensor weight;
    Tensor grad;

    static Param alloc(int d0, int d1 = 1) {
        Param p;
        p.weight = Tensor::alloc(d0, d1);
        p.grad   = Tensor::alloc(d0, d1);
        return p;
    }
    void zero_grad() { grad.zero(); }
    void release() { weight.release(); grad.release(); }
    int  size() const { return weight.size(); }
};

// Abstract model
struct Model {
    virtual ~Model() = default;
    virtual const char* name() const = 0;
    virtual int param_count() const = 0;

    // Forward: input[batch*history*N] â†’ output[batch*N]
    virtual void forward(const Tensor& input, Tensor& output,
                         int batch, int history, int width) = 0;

    // Backward: given d_loss/d_output, compute d_loss/d_params
    // Also fills d_input if needed (for stacking, not required here)
    virtual void backward(const Tensor& d_output, const Tensor& input,
                          int batch, int history, int width) = 0;

    // Access parameters for optimizer
    virtual int  num_params() const = 0;
    virtual Param& param(int idx) = 0;

    void zero_grad() {
        for (int i = 0; i < num_params(); ++i) param(i).zero_grad();
    }

    void print_summary() {
        fprintf(stderr, "  Model: %s | %d parameters\n", name(), param_count());
    }
};

} // namespace well
