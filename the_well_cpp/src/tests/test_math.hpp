// ============================================================================
// TEST_MATH.HPP — Bare-metal math validation
//
// Tests all math:: functions against known analytical values.
// Reports via serial/VGA. Pure C++, no libc.
// ============================================================================
#pragma once
#include "../runtime/bare_metal.hpp"

namespace test {

struct MathResult {
    int passed = 0;
    int failed = 0;
    
    void check(const char* name, float got, float expected, float tol = 1e-4f) {
        float err = math::abs(got - expected);
        if (err <= tol) {
            passed++;
        } else {
            print::str("  FAIL: "); print::str(name);
            print::str(" got="); print::flt(got, 6);
            print::str(" expected="); print::flt(expected, 6);
            print::str(" err="); print::flt(err, 8);
            print::line("");
            failed++;
        }
    }
};

static MathResult run_math_tests() {
    MathResult r;
    
    // ── sqrt ──
    r.check("sqrt(4)",   math::sqrt(4.0f),    2.0f);
    r.check("sqrt(9)",   math::sqrt(9.0f),    3.0f);
    r.check("sqrt(2)",   math::sqrt(2.0f),    1.41421356f);
    r.check("sqrt(0.25)", math::sqrt(0.25f),  0.5f);
    r.check("sqrt(1)",   math::sqrt(1.0f),    1.0f);
    
    // ── sin ──
    r.check("sin(0)",       math::sin(0.0f),         0.0f);
    r.check("sin(pi/2)",   math::sin(math::PI/2),   1.0f);
    r.check("sin(pi)",     math::sin(math::PI),     0.0f, 1e-3f);
    r.check("sin(3pi/2)",  math::sin(3*math::PI/2), -1.0f, 1e-3f);
    r.check("sin(pi/6)",   math::sin(math::PI/6),   0.5f, 1e-3f);
    
    // ── cos ──
    r.check("cos(0)",       math::cos(0.0f),         1.0f);
    r.check("cos(pi/2)",   math::cos(math::PI/2),   0.0f, 1e-3f);
    r.check("cos(pi)",     math::cos(math::PI),     -1.0f, 1e-3f);
    r.check("cos(pi/3)",   math::cos(math::PI/3),   0.5f, 1e-3f);
    
    // ── exp ──
    r.check("exp(0)",  math::exp(0.0f),  1.0f);
    r.check("exp(1)",  math::exp(1.0f),  2.71828183f, 1e-3f);
    r.check("exp(-1)", math::exp(-1.0f), 0.36787944f, 1e-3f);
    r.check("exp(2)",  math::exp(2.0f),  7.38905610f, 1e-2f);
    
    // ── log ──
    r.check("log(1)",      math::log(1.0f),     0.0f, 1e-3f);
    r.check("log(e)",      math::log(math::E),  1.0f, 1e-3f);
    r.check("log(e^2)",    math::log(math::E * math::E), 2.0f, 1e-2f);
    
    // ── tanh ──
    r.check("tanh(0)",    math::tanh(0.0f),    0.0f);
    r.check("tanh(big)",  math::tanh(10.0f),   1.0f, 1e-5f);
    r.check("tanh(-big)", math::tanh(-10.0f), -1.0f, 1e-5f);
    
    // ── atan2 ──
    r.check("atan2(0,1)",  math::atan2(0.0f, 1.0f),  0.0f);
    r.check("atan2(1,0)",  math::atan2(1.0f, 0.0f),  math::PI/2, 1e-3f);
    r.check("atan2(0,-1)", math::atan2(0.0f, -1.0f), math::PI, 1e-3f);
    
    // ── pow ──
    r.check("pow(2,3)",  math::pow(2.0f, 3.0f),  8.0f, 1e-2f);
    r.check("pow(3,2)",  math::pow(3.0f, 2.0f),  9.0f, 1e-2f);
    r.check("pow(2,0.5)", math::pow(2.0f, 0.5f), 1.41421356f, 1e-2f);
    
    // ── abs ──
    r.check("abs(-5)",  math::abs(-5.0f), 5.0f);
    r.check("abs(5)",   math::abs(5.0f),  5.0f);
    r.check("abs(0)",   math::abs(0.0f),  0.0f);
    
    // ── Identities ──
    // sin²x + cos²x = 1
    for (int i = 0; i < 8; ++i) {
        float x = (float)i * 0.7f;
        float s = math::sin(x), c = math::cos(x);
        r.check("sin2+cos2=1", s*s + c*c, 1.0f, 1e-3f);
    }
    
    // exp(log(x)) = x
    for (int i = 1; i <= 5; ++i) {
        float x = (float)i * 1.5f;
        r.check("exp(log(x))=x", math::exp(math::log(x)), x, 0.05f);
    }
    
    return r;
}

} // namespace test
