import torch


def get_tols(dtype: torch.dtype, op_type="elementwise"):
    """
    get tolerances according to dtype and op type

    Args:
    - dtype: torch.dtype, [torch.float32, torch.float16, torch.bfloat16]
    - op_type: str, ['elementwise', 'reduction', 'nonlinear', 'matmul']

    Returns:
    - (dict): {'rtol': value, 'atol': value}
    """

    # define tolerances dtype --> op_type --> (rtol, atol)
    tols = {
        torch.float32: {
            "elementwise": (1e-5, 1e-6),
            "reduction":   (1e-4, 1e-5),
            "nonlinear":   (1e-4, 1e-5),
            "matmul":      (1e-3, 1e-4),
        },
        torch.float16: {
            "elementwise": (1e-3, 1e-3),
            "reduction":   (5e-3, 1e-2),
            "nonlinear":   (1e-2, 1e-3),
            "matmul":      (5e-3, 1e-2),
        },
        torch.bfloat16: {
            "elementwise": (1.6e-2, 1e-2),
            "reduction":   (2e-2,  5e-2),
            "nonlinear":   (5e-2,  1e-2),
            "matmul":      (3e-2,  5e-2),
        }
    }

    if dtype not in tols:
        raise ValueError(f"Unsupported dtype: {dtype}. Use fp32, fp16, or bf16.")

    if op_type not in tols[dtype]:
        # if op_type is unknown, return the most strict elementwise tolerances
        rtol, atol = tols[dtype]["elementwise"]
    else:
        rtol, atol = tols[dtype][op_type]

    return {'rtol': rtol, 'atol': atol}
