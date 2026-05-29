
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "cuda/dense_ft_adj_mul.h"
#include "cuda/dense_ft_antipode.h"
#include "cuda/dense_ft_exp.h"
#include "cuda/dense_ft_fma.h"
#include "cuda/dense_ft_fmexp.h"
#include "cuda/dense_ft_log.h"
#include "cuda/dense_lie_pairing.h"
#include "cuda/dense_st_adj_mul.h"
#include "cuda/dense_st_fma.h"
#include "cuda/dense_tensor_pairing.h"

static inline int add_fn_capsule(PyObject* dict, const char* name, void* fn_ptr)
{
    PyObject* capsule = PyCapsule_New(fn_ptr, name, NULL);
    if (capsule == NULL) {
        return -1;
    }

    const int ret = PyDict_SetItemString(dict, name, capsule);
    Py_DECREF(capsule);

    return ret;
}

#define RPJ_ADD_FN_CAPSULE(dict, fn) add_fn_capsule(dict, #fn, (void*) fn)

static int make_jax_function_dict(PyObject* module)
{
    int ret = -1;
    PyObject* dict = PyDict_New();
    if (dict == NULL) {
        return ret;
    }

    if (RPJ_ADD_FN_CAPSULE(dict, cuda_dense_ft_antipode) < 0) {
        goto finish;
    }
    if (RPJ_ADD_FN_CAPSULE(dict, cuda_dense_ft_exp) < 0) {
        goto finish;
    }
    if (RPJ_ADD_FN_CAPSULE(dict, cuda_dense_ft_fma) < 0) {
        goto finish;
    }
    if (RPJ_ADD_FN_CAPSULE(dict, cuda_dense_ft_mul) < 0) {
        goto finish;
    }
    if (RPJ_ADD_FN_CAPSULE(dict, cuda_dense_ft_fmexp) < 0) {
        goto finish;
    }
    if (RPJ_ADD_FN_CAPSULE(dict, cuda_dense_ft_log) < 0) {
        goto finish;
    }
    if (RPJ_ADD_FN_CAPSULE(dict, cuda_dense_st_fma) < 0) {
        goto finish;
    }
    if (RPJ_ADD_FN_CAPSULE(dict, cuda_dense_st_mul) < 0) {
        goto finish;
    }
    if (RPJ_ADD_FN_CAPSULE(dict, cuda_dense_ft_adj_lmul) < 0) {
        goto finish;
    }
    if (RPJ_ADD_FN_CAPSULE(dict, cuda_dense_ft_adj_rmul) < 0) {
        goto finish;
    }
    if (RPJ_ADD_FN_CAPSULE(dict, cuda_dense_st_adj_mul) < 0) {
        goto finish;
    }

    ret = PyModule_AddObjectRef(module, "cuda_functions", dict);
finish:
    Py_DECREF(dict);
    return ret;
}

static PyMethodDef rpy_jax_cuda_methods[] = {
    {NULL, NULL, 0, NULL}
};

static PyModuleDef_Slot rpy_jax_cuda_slots[] = {
    {Py_mod_exec, make_jax_function_dict},
#if PY_VERSION_HEX >= 0x030C0000 && (!defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x030C0000)
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
    {0, NULL}
};

static struct PyModuleDef rpy_jax_cuda_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_rpy_jax_cuda",
    .m_doc = "RoughPy JAX CUDA plugin internals",
    .m_size = 0,
    .m_methods = rpy_jax_cuda_methods,
    .m_slots = rpy_jax_cuda_slots,
};

PyMODINIT_FUNC PyInit__rpy_jax_cuda(void)
{
    return PyModuleDef_Init(&rpy_jax_cuda_module);
}
